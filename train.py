import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph
import time
import os
from typing import cast

from Utils import Utils
from DownstreamModel import DownstreamModel
from GeoGNN import GeoGNNModel
from esol_dataset import ESOLDataset, ESOLDataElement


# Set to only use the 3rd GPU (ie. GPU-2).
# Since GPU-0 is over-subscribed, and also I'm told to only use 1 out of our 4 GPUs.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class GraphDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        use_gpu: bool = True
    ) -> None:
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn)
        self.use_gpu = use_gpu

    def _collate_fn(self, batch: list[ESOLDataElement]) -> tuple[DGLGraph, DGLGraph, Tensor]:
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        labels: list[Tensor] = []
        for elem in batch:
            smiles = cast(str, elem['smiles'])
            label = cast(Tensor, elem['label'])
            atom_bond_graph, bond_angle_graph = Utils.smiles_to_graphs(smiles, self.use_gpu)
            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            labels.append(label)
        return (
            dgl.batch(atom_bond_graphs),
            dgl.batch(bond_angle_graphs),
            torch.stack(labels).to('cuda' if self.use_gpu else 'cpu')
        )


def train_model(num_epochs: int = 100) -> None:
    # Use GPU if available, else use CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init / Load all the object instances.
    compound_encoder, model, criterion, data_loader, optimizer = _init_objects(device)
    compound_encoder, model, optimizer, previous_epoch, epoch_losses \
        = _load_checkpoint_if_exists('./checkpoints/', model, optimizer)

    # Train model
    start_epoch: int = previous_epoch + 1   # start from the next epoch
    start_time = time.time()
    losses: list[float] = []
    for epoch in range(start_epoch, num_epochs):
        for i, batch_data in enumerate(data_loader):
            atom_bond_graphs, bond_angle_graphs, labels = cast(tuple[DGLGraph, DGLGraph, Tensor], batch_data)

            # Zero grad the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(atom_bond_graphs, bond_angle_graphs)

            # Calculate loss
            loss = criterion.forward(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            losses.append(loss.item())
            end_time = time.time()
            print(f'Batch {i+1:04}, Time: {end_time - start_time:.2f}, Loss: {loss.item():06.3f}')
            start_time = end_time

        avg_loss = sum(losses) / len(losses)
        losses = []
        epoch_losses.append(avg_loss)
        prev_epoch_loss = epoch_losses[-2] if len(epoch_losses) >= 2 else 0.0
        print(f'=== Epoch {epoch+1:04}, Avg loss: {avg_loss:06.3f}, Prev loss: {prev_epoch_loss:06.3f} ===')

        torch.save({
            'epoch': epoch,
            'epoch_losses': epoch_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'./checkpoints/esol_only_checkpoint_{epoch}.pth')


def _init_objects(device: torch.device) \
    -> tuple[GeoGNNModel, DownstreamModel, nn.MSELoss, GraphDataLoader, Adam]:
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

    # Define loss function - since ESOL is a regression task, we use MSE loss
    criterion = nn.MSELoss()

    data_loader = GraphDataLoader(
        dataset = ESOLDataset(),
        batch_size = 32,
        shuffle = True,
        use_gpu = device.type == "cuda"
    )
    optimizer = Adam(model.parameters())

    return compound_encoder, model, criterion, data_loader, optimizer


def _load_checkpoint_if_exists(
    checkpoint_dir: str,
    model: DownstreamModel,
    optimizer: Adam
) -> tuple[GeoGNNModel, DownstreamModel, Adam, int, list[float]]:
    # Make the checkpoint dir if it doesn't exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check if there is a checkpoint
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    has_checkpoint = len(checkpoint_files) > 0
    if not has_checkpoint:
        # If not, return the arguments as is / default values for epoch/loss-list.
        return model.compound_encoder, model, optimizer, -1, []

    # load the last checkpoint
    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    previous_epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']
    print(f'Loaded checkpoint from epoch {previous_epoch}')

    return model.compound_encoder, model, optimizer, previous_epoch, epoch_losses


if __name__ == "__main__":
    train_model()
