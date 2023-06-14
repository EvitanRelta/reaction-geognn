import os, time, random, torch, dgl, numpy as np
from torch import Tensor, nn
from torch.optim import Adam
from dgl import DGLGraph
from typing import TypeAlias, TypedDict, cast

from DownstreamModel import DownstreamModel
from GeoGNN import GeoGNNModel
from geognn_datasets import GeoGNNDataLoader, ESOLDataset, ScaffoldSplitter


# Set to only use the 3rd GPU (ie. GPU-2).
# Since GPU-0 is over-subscribed, and also I'm told to only use 1 out of our 4 GPUs.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Set seed to make code deterministic.
seed = 69420
random.seed(seed)
np.random.seed(seed)
dgl.random.seed(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_training(
    encoder_lr: float,
    head_lr: float,
    dropout_rate: float,
    num_epochs: int,
    device: torch.device,
    load_save_checkpoints: bool = True,
    base_checkpoint_dir: str = './checkpoints',
) -> None:
    sub_dir_name = f'esol_only_encoder_lr{encoder_lr}_head_lr{head_lr}_dropout_rate{dropout_rate}'
    checkpoint_dir = os.path.join(base_checkpoint_dir, sub_dir_name)

    # Init / Load all the object instances.
    compound_encoder, model, criterion, metric, train_data_loader, \
        valid_data_loader, test_data_loader, encoder_optimizer, head_optimizer \
        = _init_objects(device, encoder_lr, head_lr, dropout_rate)
    previous_epoch = -1
    epoch_losses: list[float] = []
    epoch_valid_losses: list[float] = []
    epoch_test_losses: list[float] = []
    if load_save_checkpoints:
        compound_encoder, model, encoder_optimizer, head_optimizer, previous_epoch, epoch_losses \
            = _load_checkpoint_if_exists(checkpoint_dir, model, encoder_optimizer, head_optimizer)

    # Train model
    start_epoch: int = previous_epoch + 1   # start from the next epoch
    for epoch in range(start_epoch, num_epochs):
        train_loss = _train(model, criterion, train_data_loader, encoder_optimizer, head_optimizer)
        valid_loss = _evaluate(model, metric, valid_data_loader)
        test_loss = _evaluate(model, metric, test_data_loader)

        epoch_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_test_losses.append(test_loss)
        prev_epoch_loss = epoch_losses[-2] if len(epoch_losses) >= 2 else 0.0
        print(f'=== Epoch {epoch:04}, Train loss: {train_loss:06.3f}, Prev loss: {prev_epoch_loss:06.3f} (Valid | Test losses: {valid_loss:06.3f} | {test_loss:06.3f}) ===')

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


def _init_objects(
    device: torch.device,
    encoder_lr: float,
    head_lr: float,
    dropout_rate: float,
) -> tuple[GeoGNNModel, DownstreamModel, TrainCriterion, Metric, TrainDataLoader, ValidDataLoader, TestDataLoader, EncoderOptimizer, HeadOptimizer]:
    """
    Initialize all the required object instances.
    """
    # Instantiate GNN model
    compound_encoder = GeoGNNModel(dropout_rate=dropout_rate)
    model = DownstreamModel(
        compound_encoder=compound_encoder,
        task_type='regression',
        out_size=1,  # Since ESOL is a regression task with a single target value
        dropout_rate=dropout_rate,
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
        device = device
    )
    valid_data_loader = GeoGNNDataLoader(
        valid_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = 32,
        shuffle = False,  # No need to shuffle validation and test data
        device = device
    )
    test_data_loader = GeoGNNDataLoader(
        test_dataset,
        fit_mean = train_mean,
        fit_std = train_std,
        batch_size = 32,
        shuffle = False,  # No need to shuffle validation and test data
        device = device
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
    start_time = time.time()
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
        end_time = time.time()
        print(f'Batch {i+1:04}, Time: {end_time - start_time:.2f}, Loss: {loss.item():06.3f}')
        start_time = end_time

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


if __name__ == "__main__":
    # Use GPU.
    assert torch.cuda.is_available(), "No visible GPU."
    device = torch.device('cuda:0')

    # Try various learning-rates and dropout-rates, based on GeoGNN's
    # `finetune_regr.sh` script:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_regr.sh#L38-L39
    lr_pairs = [(1e-3, 1e-3), (1e-3, 4e-3), (4e-3, 4e-3), (4e-4, 4e-3)]
    dropout_rates = [0.1, 0.2]

    for encoder_lr, head_lr in lr_pairs:
        for dropout_rate in dropout_rates:
            run_training(
                encoder_lr = encoder_lr,
                head_lr = head_lr,
                dropout_rate = dropout_rate,
                device = device,
                num_epochs = 100,
            )
