import math
from abc import ABC

import torch
from base_classes import GeoGNNCacheDataModule, GeoGNNDataElement, GeoGNNGraphs
from torch.utils.data import Dataset, random_split
from typing_extensions import override

from .datasets import QM9_TASK_COL_NAMES, QM9Dataset
from .Preprocessing import Preprocessing


class _BaseDataModule(GeoGNNCacheDataModule, ABC):
    @override
    @classmethod
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs:
        return Preprocessing.smiles_to_graphs(smiles, torch.device('cpu'))


class QM9DataModule(_BaseDataModule):
    @override
    def get_dataset_splits(self) -> \
        tuple[Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement]]:
        dataset = QM9Dataset(self.task_column_name)

        # Based on `random_split` from newer versions of Pytorch:
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        train_val_test_frac = [0.8, 0.1, 0.1]
        subset_lengths: list[int] = []
        for frac in train_val_test_frac:
            n_items_in_split = int(math.floor(len(dataset) * frac))
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

        rand_gen = torch.Generator().manual_seed(self.random_split_seed)
        train_dataset, valid_dataset, test_dataset \
            = random_split(dataset, subset_lengths, generator=rand_gen)
        return train_dataset, test_dataset, valid_dataset

    def __init__(
        self,
        task_column_name: list[QM9_TASK_COL_NAMES],
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
        random_split_seed: int = 0,
    ):
        self.task_column_name = task_column_name
        self.random_split_seed = random_split_seed
        super().__init__(batch_size, shuffle, cache_path)
