from typing import TypedDict
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph

from Utils import Utils
from .esol_dataset import ESOLDataset


class GeoGNNDataElement(TypedDict):
    """A data entry for GeoGNN."""

    smiles: str
    """SMILES string of the data's molecule."""

    data: Tensor
    """Ground truth data. Size `(num_of_feats, num_of_entries)`"""


class GeoGNNDataLoader(DataLoader[GeoGNNDataElement]):
    """
    Data loader for GeoGNN's datasets.
    """
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu')
    ) -> None:
        dataset = ESOLDataset()
        super().__init__(dataset, batch_size, shuffle, collate_fn=self._collate_fn)
        self.device = device

    def _collate_fn(self, batch: list[GeoGNNDataElement]) -> tuple[DGLGraph, DGLGraph, Tensor]:
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
