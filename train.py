import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph
import time
import os
from typing import TypeAlias, TypedDict, cast

from Utils import Utils
from DownstreamModel import DownstreamModel
from GeoGNN import GeoGNNModel
from esol_dataset import ESOLDataset, ESOLDataElement


# Set to only use the 3rd GPU (ie. GPU-2).
# Since GPU-0 is over-subscribed, and also I'm told to only use 1 out of our 4 GPUs.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train_model(num_epochs: int = 100) -> None:
    # Use GPU if available, else use CPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Init / Load all the object instances.
    compound_encoder, model, criterion, data_loader, encoder_optimizer, head_optimizer \
        = _init_objects(device)
    previous_epoch = -1
    epoch_losses: list[float] = []
    compound_encoder, model, encoder_optimizer, head_optimizer, previous_epoch, epoch_losses \
        = _load_checkpoint_if_exists('./checkpoints/', model, encoder_optimizer, head_optimizer)

    # Train model
    start_epoch: int = previous_epoch + 1   # start from the next epoch
    start_time = time.time()
    losses: list[float] = []
    for epoch in range(start_epoch, num_epochs):
        for i, batch_data in enumerate(data_loader):
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
        losses = []
        epoch_losses.append(avg_loss)
        prev_epoch_loss = epoch_losses[-2] if len(epoch_losses) >= 2 else 0.0
        print(f'=== Epoch {epoch:04}, Avg loss: {avg_loss:06.3f}, Prev loss: {prev_epoch_loss:06.3f} ===')

        # Save checkpoint of epoch.
        checkpoint_dict: GeoGNNCheckpoint = {
            'epoch': epoch,
            'epoch_losses': epoch_losses,
            'model_state_dict': model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'head_optimizer_state_dict': head_optimizer.state_dict()
        }
        torch.save(checkpoint_dict, f'./checkpoints/esol_only_checkpoint_{epoch}.pth')





# ==================================================
#          Helper types/classes/functions
# ==================================================
Criterion: TypeAlias = nn.L1Loss
EncoderOptimizer: TypeAlias = Adam
HeadOptimizer: TypeAlias = Adam

class GeoGNNCheckpoint(TypedDict):
    """Dict type of a loaded checkpoint."""
    
    epoch: int
    """Epoch for this checkpoint (zero-indexed)."""

    epoch_losses: list[float]
    """Losses for each epoch."""

    model_state_dict: dict
    """State dict of the `DownstreamModel` instance."""

    encoder_optimizer_state_dict: dict
    """State dict of the `Adam` optimizer for the `GeoGNN` instance."""

    head_optimizer_state_dict: dict
    """
    State dict of the `Adam` optimizer for the `DownstreamModel` parameters but
    excluding those in `GeoGNN`.
    """

class GeoGNNDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn)
        self.device = device

    def _collate_fn(self, batch: list[ESOLDataElement]) -> tuple[DGLGraph, DGLGraph, Tensor]:
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        data_list: list[Tensor] = []
        for elem in batch:
            smiles, data = elem['smiles'], elem['data']
            atom_bond_graph, bond_angle_graph = Utils.smiles_to_graphs(smiles, self.device)
            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            data_list.append(data)

        return (
            dgl.batch(atom_bond_graphs),
            dgl.batch(bond_angle_graphs),
            torch.stack(data_list).to(self.device)
        )

def _init_objects(device: torch.device) \
    -> tuple[GeoGNNModel, DownstreamModel, Criterion, GeoGNNDataLoader, EncoderOptimizer, HeadOptimizer]:
    """
    Initialize all the required object instances.
    """
    # Instantiate GNN model
    compound_encoder = GeoGNNModel()
    model = DownstreamModel(
        compound_encoder=compound_encoder,
        task_type='regression',
        out_size=1  # Since ESOL is a regression task with a single target value
    )
    model = model.to(device)

    # Loss function based on GeoGNN's `finetune_regr.py`:
    # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/finetune_regr.py#L159-L160
    criterion = torch.nn.L1Loss()

    data_loader = GeoGNNDataLoader(
        dataset = ESOLDataset(),
        batch_size = 32,
        shuffle = True,
        device = device
    )

    # DownstreamModel params but excluding those in GeoGNN.
    head_params = list(set(model.parameters()) - set(compound_encoder.parameters()))

    encoder_optimizer = Adam(compound_encoder.parameters())
    head_optimizer = Adam(head_params)

    return compound_encoder, model, criterion, data_loader, encoder_optimizer, head_optimizer


def _load_checkpoint_if_exists(
    checkpoint_dir: str,
    model: DownstreamModel,
    encoder_optimizer: Adam,
    head_optimizer: Adam
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


if __name__ == "__main__":
    train_model()
