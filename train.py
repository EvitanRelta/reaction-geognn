import os, time, random, torch, dgl, numpy as np, subprocess, pickle, argparse
from tqdm.autonotebook import tqdm
from torch import Tensor, nn
from torch.optim import Adam
from dgl import DGLGraph
from typing import TypeAlias, TypedDict, cast

from DownstreamModel import DownstreamModel
from GeoGNN import GeoGNNModel
from geognn_datasets import GeoGNNDataLoader, ESOLDataset, ScaffoldSplitter, GeoGNNDataset
from Preprocessing import Preprocessing


# Set seed to make code deterministic.
SEED = 69420
random.seed(SEED)
np.random.seed(SEED)
dgl.random.seed(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    num_downstream_mlp_layers: int,
    dropout_rate: float,
    num_epochs: int,
    device: torch.device,
    load_save_checkpoints: bool = True,
    base_checkpoint_dir: str = './checkpoints',
) -> None:
    sub_dir_name = f'esol_only_encoder_lr{encoder_lr}_head_lr{head_lr}_dropout_rate{dropout_rate}_mlp{num_downstream_mlp_layers}'
    checkpoint_dir = os.path.join(base_checkpoint_dir, sub_dir_name)

    # Init / Load all the object instances.
    compound_encoder, model, criterion, metric, train_data_loader, \
        valid_data_loader, test_data_loader, encoder_optimizer, head_optimizer \
        = _init_objects(device, encoder_lr, head_lr, num_downstream_mlp_layers, dropout_rate)
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


TrainCriterion: TypeAlias = nn.L1Loss
Metric: TypeAlias = RMSELoss
EncoderOptimizer: TypeAlias = Adam
HeadOptimizer: TypeAlias = Adam
TrainDataLoader: TypeAlias = GeoGNNDataLoader
ValidDataLoader: TypeAlias = GeoGNNDataLoader
TestDataLoader: TypeAlias = GeoGNNDataLoader


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
    dataset: GeoGNNDataset,
    save_file_path: str,
    device: torch.device,
) -> dict[str, tuple[DGLGraph, DGLGraph]]:
    # Extract all the SMILES strings from dataset.
    smiles_list = [data['smiles'] for data in dataset.data_list]

    # If the save file exists, load the graphs from it.
    if os.path.exists(save_file_path):
        print(f'Loading cached graphs file at "{save_file_path}"...\n')
        cached_graphs = _load_cached_graphs(save_file_path, device)

        assert len(cached_graphs) == len(dataset), \
            "Length of saved cached-graphs doesn't match dataset length."
        assert set(cached_graphs.keys()) == set(smiles_list), \
            "SMILES of saved cached-graphs doesn't match those in the dataset."

        return cached_graphs

    # If the save file doesn't exist, compute all the graphs
    cached_graphs = _compute_all_graphs(smiles_list, device)

    # Create the parent directory of the save file path if it doesn't exist
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    # Save the graphs to the save file path
    _save_cached_graphs(cached_graphs, save_file_path)

    return cached_graphs

def _compute_all_graphs(
    smiles_list: list[str],
    device: torch.device,
) -> dict[str, tuple[DGLGraph, DGLGraph]]:
    precomputed_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {}
    print(f'Precomputing graphs for {len(smiles_list)} SMILES strings:')
    for smiles in tqdm(smiles_list):
        precomputed_graphs[smiles] = Preprocessing.smiles_to_graphs(smiles, device=device)
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
    num_downstream_mlp_layers: int,
    dropout_rate: float,
) -> tuple[GeoGNNModel, DownstreamModel, TrainCriterion, Metric, TrainDataLoader, ValidDataLoader, TestDataLoader, EncoderOptimizer, HeadOptimizer]:
    """
    Initialize all the required object instances.
    """
    # Instantiate GNN model
    compound_encoder = GeoGNNModel(dropout_rate=dropout_rate)
    model = DownstreamModel(
        compound_encoder = compound_encoder,
        task_type = 'regression',
        out_size = 1,  # Since ESOL is a regression task with a single target value
        num_of_mlp_layers = num_downstream_mlp_layers,
        dropout_rate = dropout_rate,
    )
    model = model.to(device)

    # Loss function based on GeoGNN's `finetune_regr.py`:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/finetune_regr.py#L159-L160
    criterion = torch.nn.L1Loss()

    # Loss function for evaluating against validation/test datasets,
    # based on GeoGNN's `finetune_regr.py` metric for `esol` dataset:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/finetune_regr.py#L103-L104
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/finetune_regr.py#L117-L118
    metric = RMSELoss()

    # Get and split dataset.
    dataset = ESOLDataset()
    cached_graphs = _get_cached_graphs(
        dataset,
        save_file_path = './cached_graphs/cached_esol_graphs.bin',
        device = device,
    )
    train_dataset, valid_dataset, test_dataset \
        = ScaffoldSplitter().split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    # Defined data-loader, where the data is standardize with the
    # training mean and standard deviation.
    train_mean, train_std = GeoGNNDataLoader.get_stats(train_dataset)
    train_data_loader = GeoGNNDataLoader(
        train_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = 32,
        shuffle = True,
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )
    valid_data_loader = GeoGNNDataLoader(
        valid_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = 32,
        shuffle = False,  # No need to shuffle validation and test data
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )
    test_data_loader = GeoGNNDataLoader(
        test_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = 32,
        shuffle = False,  # No need to shuffle validation and test data
        device = device,
        cached_graphs = cached_graphs,
        worker_init_fn=_dataloader_worker,
        generator=_get_dataloader_generator(),
    )

    compound_encoder_params = list(compound_encoder.parameters())
    model_params = list(model.parameters())

    # DownstreamModel params but excluding those in GeoGNN.
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
    model: DownstreamModel,
    encoder_optimizer: Adam,
    head_optimizer: Adam,
) -> tuple[GeoGNNModel, DownstreamModel, EncoderOptimizer, HeadOptimizer, int, list[float]]:
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
    checkpoint = cast(GeoGNNCheckpoint, checkpoint)

    # Load the saved values in the checkpoint.
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    head_optimizer.load_state_dict(checkpoint['head_optimizer_state_dict'])
    previous_epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']
    print(f'Loaded checkpoint from epoch {previous_epoch}')

    return model.compound_encoder, model, encoder_optimizer, head_optimizer, previous_epoch, epoch_losses


def _train(
    model: DownstreamModel,
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
        batch_atom_bond_graph, batch_bond_angle_graph, labels \
            = cast(tuple[DGLGraph, DGLGraph, Tensor], batch_data)

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
    model: DownstreamModel,
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
            batch_atom_bond_graph, batch_bond_angle_graph, labels \
                = cast(tuple[DGLGraph, DGLGraph, Tensor], batch_data)
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

    # Try various learning-rates, dropout-rates and layers of
    # `DownstreamModel` MLP, based on GeoGNN's `finetune_regr.sh` script:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_regr.sh#L37-L39
    lr_pairs = [(1e-3, 1e-3), (1e-3, 4e-3), (4e-3, 4e-3), (4e-4, 4e-3)]
    dropout_rates = [0.1, 0.2]
    downstream_mlp_layers_list = [2, 3]

    for encoder_lr, head_lr in lr_pairs:
        for dropout_rate in dropout_rates:
            for num_downstream_mlp_layers in downstream_mlp_layers_list:
                run_training(
                    encoder_lr = encoder_lr,
                    head_lr = head_lr,
                    num_downstream_mlp_layers = num_downstream_mlp_layers,
                    dropout_rate = dropout_rate,
                    device = device,
                    num_epochs = 100,
                    load_save_checkpoints = args_dict['load_save_checkpoints']
                )
