from abc import ABC

import torch
from dgl import DGLGraph
from lightning_utils import GeoGNNCacheDataModule
from torch.utils.data import Dataset
from typing_extensions import override

from .datasets import QM9_TASK_COL_NAMES, GeoGNNDataElement, QM9Dataset, \
    ScaffoldSplitter
from .Preprocessing import Preprocessing


class _BaseDataModule(GeoGNNCacheDataModule, ABC):
    @override
    @classmethod
    def compute_graphs(cls, smiles: str) -> tuple[DGLGraph, DGLGraph]:
        return Preprocessing.smiles_to_graphs(smiles, torch.device('cpu'))


class QM9DataModule(_BaseDataModule):
    @override
    def get_dataset_splits(self) -> \
        tuple[Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement]]:
        dataset = QM9Dataset(self.task_column_name)
        train_dataset, valid_dataset, test_dataset \
            = ScaffoldSplitter().split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        return train_dataset, test_dataset, valid_dataset

    def __init__(
        self,
        task_column_name: list[QM9_TASK_COL_NAMES],
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
    ):
        self.task_column_name = task_column_name
        super().__init__(batch_size, shuffle, cache_path)
