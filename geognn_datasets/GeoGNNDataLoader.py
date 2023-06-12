import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph

from Utils import Utils
from esol_dataset import ESOLDataset, ESOLDataElement


class GeoGNNDataLoader(DataLoader):
    """
    Data loader for GeoGNN's datasets.
    """
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
