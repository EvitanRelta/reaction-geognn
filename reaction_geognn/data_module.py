from typing import Literal

import torch
from base_classes import GeoGNNCacheDataModule, GeoGNNDataset, GeoGNNGraphs
from typing_extensions import override

from .datasets import get_wb97_fold_dataset
from .graph_utils import superimpose_reactant_products_graphs
from .preprocessing import reaction_smart_to_graph


class Wb97DataModule(GeoGNNCacheDataModule):
    @override
    def get_dataset_splits(self) -> tuple[GeoGNNDataset, GeoGNNDataset, GeoGNNDataset]:
        return get_wb97_fold_dataset(self.fold_num)

    @override
    @classmethod
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs:
        atom_bond_graph, bond_angle_graph = reaction_smart_to_graph(smiles, torch.device('cpu'))
        return (
            atom_bond_graph,
            bond_angle_graph,
            superimpose_reactant_products_graphs(atom_bond_graph),
        )

    def __init__(
        self,
        fold_num: Literal[0, 1, 2, 3, 4],
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
    ):
        self.fold_num: Literal[0, 1, 2, 3, 4] = fold_num
        super().__init__(batch_size, shuffle, cache_path)


class B97DataModule(Wb97DataModule):
    @override
    def get_dataset_splits(self) -> tuple[GeoGNNDataset, GeoGNNDataset, GeoGNNDataset]:
        wb97, b97 = get_wb97_fold_dataset(self.fold_num, include_pretrain=True)
        return b97
