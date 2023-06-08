import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph
import time

from Utils import Utils
from DownstreamModel import DownstreamModel
from GeoGNN import GeoGNNModel
from esol_dataset import load_esol_dataset, ESOLDataElement
from typing import cast


class ESOLDataset(Dataset):
    def __init__(self) -> None:
        self.dataset, self.mean, self.std = load_esol_dataset()

        # Temp fix for dataset. Removes a data element with SMILES "C" which was
        # crashing the code.
        self.dataset.pop(934)
        self.mean = self.mean[torch.arange(len(self.mean)) != 934]
        self.std = self.std[torch.arange(len(self.std)) != 934]

    def __getitem__(self, index: int) -> ESOLDataElement:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


class GraphDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        super().__init__(dataset, batch_size, shuffle, collate_fn=GraphDataLoader.collate_fn)

    @staticmethod
    def collate_fn( # type: ignore
        batch: list[ESOLDataElement]
    ) -> tuple[DGLGraph, DGLGraph, Tensor]:
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        labels: list[Tensor] = []
        for elem in batch:
            smiles = cast(str, elem['smiles'])
            label = cast(Tensor, elem['label'])
            atom_bond_graph, bond_angle_graph = Utils.smiles_to_graphs(smiles)
            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            labels.append(label)
        return dgl.batch(atom_bond_graphs), dgl.batch(bond_angle_graphs), torch.stack(labels)


def train_model(num_epochs: int = 100) -> None:
    # Instantiate your GNN model
    compound_encoder = GeoGNNModel()  # Add necessary arguments if needed

    # Define DownstreamModel
    model = DownstreamModel(
        compound_encoder=compound_encoder,
        task_type='regression',
        out_size=1  # Since ESOL is a regression task with a single target value
        # Add other arguments if needed
    )

    # If you have GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create a DataLoader
    data_loader = GraphDataLoader(ESOLDataset(), batch_size=32, shuffle=True)

    # Define your optimizer
    optimizer = Adam(model.parameters())

    # Define loss function - since ESOL is a regression task, we use MSE loss
    criterion = torch.nn.MSELoss()

    # Train model
    start_time = time.time()
    losses: list[float] = []
    for epoch in range(num_epochs):
        for i, (atom_bond_graphs, bond_angle_graphs, labels) in enumerate(data_loader):
            atom_bond_graphs = atom_bond_graphs.to(device)
            bond_angle_graphs = bond_angle_graphs.to(device)
            labels = labels.to(device)

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
        print(f'=== Epoch {epoch+1:04}, Avg loss: {avg_loss:06.3f} ===')
        torch.save(model.state_dict(), f'./checkpoints/esol_only_checkpoint_{epoch}.pth')


if __name__ == "__main__":
    train_model()
