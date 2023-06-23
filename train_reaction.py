import argparse, os, pickle, random, subprocess, time
from typing import Literal, TypeAlias, TypedDict

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.datasets import GeoGNNDataElement, GeoGNNDataset
from reaction_geognn.datasets import get_wb97_fold_dataset
from reaction_geognn.model import ProtoDataLoader, ProtoModel
from reaction_geognn.preprocessing import reaction_smart_to_graph
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

# Set seed to make code deterministic.
# Seed 0 based on chemprop's default seed:
# https://github.com/chemprop/chemprop/blob/0c3f334/README.md#trainvalidationtest-splits
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
dgl.random.seed(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True   # type: ignore
torch.backends.cudnn.benchmark = False      # type: ignore

# Functions for preserving reproducibility in PyTorch's `DataLoader`.
def _dataloader_worker(worker_id: int) -> None:
    np.random.seed(SEED)
    random.seed(SEED)

def _get_dataloader_generator() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


def run_training(
    encoder_lr: float,
    head_lr: float,
    dropout_rate: float,
    fold_num: Literal[0, 1, 2, 3, 4],
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    load_save_checkpoints: bool = True,
    base_checkpoint_dir: str = './checkpoints/reaction_geognn',
) -> None:
    sub_dir_name = f'/encoderlr{encoder_lr}_headlr{head_lr}_dropout{dropout_rate}_batchsize{batch_size}/fold_{fold_num}'
    checkpoint_dir = os.path.join(base_checkpoint_dir, sub_dir_name)

    # Init / Load all the object instances.
    compound_encoder, model, criterion, metric, train_data_loader, \
        valid_data_loader, test_data_loader, encoder_optimizer, head_optimizer \
        = _init_objects(device, encoder_lr, head_lr, dropout_rate, fold_num, batch_size)
    previous_epoch = -1
    epoch_losses: list[float] = []
    epoch_valid_losses: list[float] = []
    epoch_test_losses: list[float] = []
    if load_save_checkpoints:
        compound_encoder, model, encoder_optimizer, head_optimizer, previous_epoch, epoch_losses \
            = _load_checkpoint_if_exists(checkpoint_dir, model, encoder_optimizer, head_optimizer)

    # Train model
    start_time = time.time()
    start_epoch: int = previous_epoch + 1   # start from the next epoch
    for epoch in range(start_epoch, num_epochs):
        train_loss = _train(model, criterion, train_data_loader, encoder_optimizer, head_optimizer)
        valid_loss = _evaluate(model, metric, valid_data_loader)
        test_loss = _evaluate(model, metric, test_data_loader)

        epoch_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_test_losses.append(test_loss)
        prev_epoch_loss = epoch_losses[-2] if len(epoch_losses) >= 2 else 0.0

        end_time = time.time()
        print(f'Epoch {epoch:04}, Time: {end_time - start_time:.2f}, Train loss: {train_loss:06.3f}, Prev loss: {prev_epoch_loss:06.3f} (Valid | Test losses: {valid_loss:06.3f} | {test_loss:06.3f})')
        start_time = end_time

        if load_save_checkpoints:
            # Save checkpoint of epoch.
            checkpoint_dict: GeoGNNCheckpoint = {
                'epoch': epoch,
                'epoch_losses': epoch_losses,
                'epoch_valid_losses': epoch_valid_losses,
                'epoch_test_losses': epoch_test_losses,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'head_optimizer_state_dict': head_optimizer.state_dict()
            }
            checkpoint_filename = f'epoch_{epoch}.pth'
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, checkpoint_filename))





# ==================================================
#          Helper types/classes/functions
# ==================================================
class RMSELoss(nn.Module):
    """
    Criterion that measures the root mean squared error.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse_loss = nn.functional.mse_loss(input, target)
        return torch.sqrt(mse_loss)


TrainCriterion: TypeAlias = nn.MSELoss
Metric: TypeAlias = RMSELoss
EncoderOptimizer: TypeAlias = Adam
HeadOptimizer: TypeAlias = Adam
TrainDataLoader: TypeAlias = ProtoDataLoader
ValidDataLoader: TypeAlias = ProtoDataLoader
TestDataLoader: TypeAlias = ProtoDataLoader


class GeoGNNCheckpoint(TypedDict):
    """Dict type of a loaded checkpoint."""

    epoch: int
    """Epoch for this checkpoint (zero-indexed)."""

    epoch_losses: list[float]
    """Losses for each epoch."""

    epoch_valid_losses: list[float]
    """Validation losses for each epoch."""

    epoch_test_losses: list[float]
    """Test losses for each epoch."""

    model_state_dict: dict
    """State dict of the `DownstreamModel` instance."""

    encoder_optimizer_state_dict: dict
    """State dict of the `Adam` optimizer for the `GeoGNN` instance."""

    head_optimizer_state_dict: dict
    """
    State dict of the `Adam` optimizer for the `DownstreamModel` parameters but
    excluding those in `GeoGNN`.
    """


def _get_least_utilized_and_allocated_gpu() -> torch.device:
    """
    Returns the `torch.device` of the GPU with the lowest utilization and memory
    allocation.

    Returns:
        torch.device: The `torch.device` of the least utilized and memory allocated GPU.
    """
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,utilization.gpu,memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # GPU stats are returned in separate lines
    gpu_stats = result.strip().split('\n')
    assert len(gpu_stats) > 0, "No visible GPU."
    assert len(gpu_stats) > 1, "Only 1 GPU (expected to run on a machine with multiple GPUs)."

    parsed_stats: list[tuple[int, int, int]] = []
    for gpu_stat in gpu_stats:
        stats = gpu_stat.split(', ')
        gpu_id = int(stats[0])
        utilization = int(stats[1])
        memory = int(stats[2])
        parsed_stats.append((gpu_id, utilization, memory))

        # Printing GPU stats for debugging.
        print(f'GPU-{gpu_id}: Util = {utilization}, Mem alloc = {memory}')

    # Sort GPUs by utilization first, then memory allocation.
    sorted_gpus = sorted(parsed_stats, key=lambda x: (x[1], x[2]))

    least_used_gpu_id = sorted_gpus[0][0]
    print(f'Using GPU-{least_used_gpu_id}...\n')
    return torch.device(f'cuda:{least_used_gpu_id}')


# ==================================================
#           Precomputing/Caching of graphs
# ==================================================
def _get_cached_graphs(
    datasets: list[GeoGNNDataset],
    save_file_path: str,
    device: torch.device,
) -> dict[str, tuple[DGLGraph, DGLGraph]]:
    # Extract all the SMILES strings from datasets.
    smiles_set: set[str] = set()
    for dataset in datasets:
        smiles_set.update({ data['smiles'] for data in dataset.data_list })

    # If the save file exists, load the graphs from it.
    if os.path.exists(save_file_path):
        print(f'Loading cached graphs file at "{save_file_path}"...\n')
        cached_graphs = _load_cached_graphs(save_file_path, device)

        assert len(cached_graphs) == len(smiles_set), \
            "Length of saved cached-graphs doesn't match datasets' total length."
        assert set(cached_graphs.keys()) == smiles_set, \
            "SMILES of saved cached-graphs doesn't match those in the dataset."

        return cached_graphs

    # If the save file doesn't exist, compute all the graphs
    cached_graphs = _compute_all_graphs(smiles_set, device)

    # Create the parent directory of the save file path if it doesn't exist
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    # Save the graphs to the save file path
    _save_cached_graphs(cached_graphs, save_file_path)

    return cached_graphs

def _compute_all_graphs(
    smiles_set: set[str],
    device: torch.device,
) -> dict[str, tuple[DGLGraph, DGLGraph]]:
    precomputed_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {}
    print(f'Precomputing graphs for {len(smiles_set)} SMILES strings:')
    for smiles in tqdm(smiles_set):
        precomputed_graphs[smiles] = reaction_smart_to_graph(smiles, device=device)
    print('\n')
    return precomputed_graphs

def _save_cached_graphs(
    cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]],
    save_file_path: str,
) -> None:
    with open(save_file_path, 'wb') as f:
        pickle.dump(cached_graphs, f)

def _load_cached_graphs(save_file_path: str, device: torch.device) -> dict[str, tuple[DGLGraph, DGLGraph]]:
    cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]]
    with open(save_file_path, 'rb') as f:
        cached_graphs = pickle.load(f)

    # Move all graphs to device.
    cached_graphs = {
        smiles: (g1.to(device), g2.to(device)) \
            for smiles, (g1, g2) in cached_graphs.items()
    }
    return cached_graphs
# ==================================================
# ==================================================


def _init_objects(
    device: torch.device,
    encoder_lr: float,
    head_lr: float,
    dropout_rate: float,
    fold_num: Literal[0, 1, 2, 3, 4],
    batch_size: int,
) -> tuple[GeoGNNModel, ProtoModel, TrainCriterion, Metric, TrainDataLoader, ValidDataLoader, TestDataLoader, EncoderOptimizer, HeadOptimizer]:
    """
    Initialize all the required object instances.
    """
    # Instantiate GNN model
    compound_encoder = GeoGNNModel(dropout_rate=dropout_rate)
    model = ProtoModel(
        compound_encoder = compound_encoder,
        out_size = 1,
        dropout_rate = dropout_rate,
    )
    model = model.to(device)

    # Based on the default Chemprop regression loss (ie. mse) as defined here:
    # https://github.com/chemprop/chemprop/blob/0c3f334/README.md#loss-functions
    criterion = nn.MSELoss()

    # Based on the default Chemprop regression metric (ie. rmse) as defined here:
    # https://github.com/chemprop/chemprop/blob/0c3f334/README.md#metrics
    metric = RMSELoss()

    # Get already-split dataset.
    train_dataset, valid_dataset, test_dataset = get_wb97_fold_dataset(fold_num)

    # Get/Compute graph cache.
    cached_graphs = _get_cached_graphs(
        [train_dataset, valid_dataset, test_dataset], # type: ignore
        save_file_path = './cached_graphs/cached_wb97.bin',
        device = device,
    )

    # Defined data-loader, where the data is standardize with the
    # training mean and standard deviation.
    train_mean, train_std = ProtoDataLoader.get_stats(train_dataset)
    train_data_loader = ProtoDataLoader(
        train_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = batch_size,
        shuffle = False, # No shuffling to ensure reproducibility.
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )
    valid_data_loader = ProtoDataLoader(
        valid_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = batch_size,
        shuffle = False,  # No need to shuffle validation and test data
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )
    test_data_loader = ProtoDataLoader(
        test_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = batch_size,
        shuffle = False,  # No need to shuffle validation and test data
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )

    compound_encoder_params = list(compound_encoder.parameters())
    model_params = list(model.parameters())

    # Head-model params, excluding those in GeoGNNModel.
    # `head_params` is the difference in elements between `model_params` &
    # `compound_encoder_params`.
    is_in = lambda x, lst: any(element is x for element in lst)
    head_params = [p for p in model_params if not is_in(p, compound_encoder_params)]

    encoder_optimizer = Adam(compound_encoder_params, lr=encoder_lr)
    head_optimizer = Adam(head_params, lr=head_lr)

    return compound_encoder, model, criterion, metric, train_data_loader, \
        valid_data_loader, test_data_loader, encoder_optimizer, head_optimizer


def _load_checkpoint_if_exists(
    checkpoint_dir: str,
    model: ProtoModel,
    encoder_optimizer: Adam,
    head_optimizer: Adam,
) -> tuple[GeoGNNModel, ProtoModel, EncoderOptimizer, HeadOptimizer, int, list[float]]:
    # Make the checkpoint dir if it doesn't exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check if there is a checkpoint
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    has_checkpoint = len(checkpoint_files) > 0
    if not has_checkpoint:
        # If not, return the arguments as is / default values for epoch/loss-list.
        return model.compound_encoder, model, encoder_optimizer, head_optimizer, -1, []

    # Load the last checkpoint.
    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))

    # Load the saved values in the checkpoint.
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    head_optimizer.load_state_dict(checkpoint['head_optimizer_state_dict'])
    previous_epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']
    print(f'Loaded checkpoint from epoch {previous_epoch}')

    return model.compound_encoder, model, encoder_optimizer, head_optimizer, previous_epoch, epoch_losses


def _train(
    model: ProtoModel,
    criterion: TrainCriterion,
    train_data_loader: TrainDataLoader,
    encoder_optimizer: EncoderOptimizer,
    head_optimizer: HeadOptimizer,
) -> float:
    """
    Trains `model` for 1 epoch.

    Returns:
        float: Average loss of all the batches in `data_loader` for this epoch of training.
    """
    losses: list[float] = []
    for i, batch_data in enumerate(train_data_loader):
        batch_atom_bond_graph, batch_bond_angle_graph, labels = batch_data

        # Zero grad the optimizers
        encoder_optimizer.zero_grad()
        head_optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(batch_atom_bond_graph, batch_bond_angle_graph)

        # Calculate loss
        loss = criterion.forward(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        encoder_optimizer.step()
        head_optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def _evaluate(
    model: ProtoModel,
    metric: Metric,
    data_loader: ValidDataLoader | TestDataLoader,
) -> float:
    """
    Evaluates `model` with the loss function `metric` against the dataset
    provided by `data_loader`.

    Returns:
        float: Average loss of all the batches in `data_loader`.
    """
    # Set model to "evaluate mode", disabling all the dropout layers.
    model.eval()

    with torch.no_grad():  # No need to track gradients in evaluation mode
        losses = []
        for i, batch_data in enumerate(data_loader):
            batch_atom_bond_graph, batch_bond_angle_graph, labels = batch_data
            outputs = model.forward(batch_atom_bond_graph, batch_bond_angle_graph)

            # Since the model is trained on standardized training data,
            # it'll thus output standardized values too.
            # Thus, we need to reverse the standardization both from the output and the
            # data loader (which auto-standardizes to the training data's mean/std)
            outputs = data_loader.unstandardize_data(outputs)
            labels = data_loader.unstandardize_data(labels)

            loss = metric.forward(outputs, labels)
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)

    # Switch back to training mode.
    model.train()
    return avg_loss

class Arguments(TypedDict):
    load_save_checkpoints: bool

def _parse_script_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--no-load-save', default=False, action='store_true', help='prevents loading/saving of checkpoints')
    args = parser.parse_args()

    output: Arguments = {
        'load_save_checkpoints': not args.no_load_save
    }
    if not output['load_save_checkpoints']:
        print('Warning: No loading/saving of checkpoints will be done.\n')
    return output


if __name__ == "__main__":
    args_dict = _parse_script_args()

    # Use GPU.
    assert torch.cuda.is_available(), "No visible GPU."
    assert torch.cuda.device_count() > 1, "Only 1 GPU (expected multiple GPUs)."
    device = _get_least_utilized_and_allocated_gpu()

    for fold_num in range(0, 5):
        run_training(
            encoder_lr = 1e-3,
            head_lr = 1e-3,
            dropout_rate = 0,
            fold_num = fold_num, # type: ignore
            device = device,
            num_epochs = 100,
            batch_size = 50,
            load_save_checkpoints = args_dict['load_save_checkpoints']
        )
