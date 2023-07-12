"""Base PyTorch-Lightning data-module."""

import os, pickle
from abc import ABC, abstractmethod
from itertools import chain
from typing import Literal

import dgl
import lightning.pytorch as pl
import torch
from dgl import DGLGraph
from geognn_base_classes import GeoGNNBatch, GeoGNNDataElement, \
    GeoGNNDataLoader
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from .scaler import StandardizeScaler


class GeoGNNCacheDataModule(ABC, pl.LightningDataModule):
    """Abstract base class for PyTorch-Lightning data-modules for GeoGNN
    datasets/dataloaders.

    Implements:
    - Caching of atom-bond and bond-angle graphs.
    - Standardization of dataset labels using `StandardizeScaler` via `self.scaler`.
    - Converting SMILES strings to graphs via the abstract class-method
    `self.compute_graphs`, and collating them into batches of type `GeoGNNBatch`.

    Requires the below 2 abstract methods to be implemented:

    ```python
    @abstractmethod
    def get_dataset_splits(self) -> tuple[GeoGNNDataset, GeoGNNDataset, GeoGNNDataset]: ...

    @abstractmethod
    @classmethod
    def compute_graphs(cls, smiles: str) -> tuple[DGLGraph, DGLGraph]: ...
    ```
    """

    @abstractmethod
    def get_dataset_splits(self) -> \
        tuple[Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement]]:
        """Get train, test and validation dataset splits.

        Returns:
            tuple[Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement], Dataset[GeoGNNDataElement]]: \
                Train, test and validation dataset splits in the form `(train, test, val)`.
        """

    @classmethod
    @abstractmethod
    def compute_graphs(cls, smiles: str) -> tuple[DGLGraph, DGLGraph]:
        """Compute GeoGNN's atom-bond graph and bond-angle graph from a
        molecule's/reaction's SMILES/SMART string.

        Args:
            smiles (str): Molecule's/Reaction's SMILES/SMART string.

        Returns:
            tuple[DGLGraph, DGLGraph]: Atom-bond and bond-angle graphs in the \
                form `(atom_bond_graph, bond_angle_graph)`.
        """

    def __init__(self, batch_size: int, shuffle: bool = False, cache_path: str | None = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_path = cache_path
        self._cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {}

        self.train_dataset, self.test_dataset, self.val_dataset \
            = self.get_dataset_splits()

        train_labels = torch.stack([el['data'] for el in self.train_dataset])
        self.scaler = StandardizeScaler()
        """Scaler used to transform the labels for train/val/test dataloaders."""
        self.scaler.fit(train_labels)

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        if not self.cache_path:
            return

        if os.path.exists(self.cache_path):
            self._load_cached_graphs()
            return

        self._precompute_all_graphs()
        self._save_cached_graphs()

    def _load_cached_graphs(self) -> None:
        assert self.cache_path and os.path.exists(self.cache_path)

        # Load cached graphs dict from disk.
        print(f'Loading cached graphs file from "{self.cache_path}"...\n')
        with open(self.cache_path, 'rb') as f:
            self._cached_graphs = pickle.load(f)
        print(f'Loaded cached graphs file from "{self.cache_path}".\n')

        # Check if the SMILES in the loaded dict matches that in the full dataset.
        assert set(self._cached_graphs.keys()) == {
            data['smiles'] for data in \
                chain(iter(self.train_dataset), iter(self.test_dataset), iter(self.val_dataset))
        }, f'SMILES in "{self.cache_path}" cache file doesn\'t match those in the dataset.'

    def _save_cached_graphs(self) -> None:
        assert self.cache_path

        # Create the parent directory of the cache path if it doesn't exist.
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Save cached graphs dict to disk.
        print(f'Saving cached graphs to "{self.cache_path}"...\n')
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self._cached_graphs, f)
        print(f'Saved cached graphs to "{self.cache_path}".\n')

    def _precompute_all_graphs(self) -> None:
        full_smiles_set: set[str] = {
            data['smiles'] for data in \
                chain(iter(self.train_dataset), iter(self.test_dataset), iter(self.val_dataset))
        }
        print(f'Precomputing graphs for {len(full_smiles_set)} SMILES strings:')
        for smiles in tqdm(full_smiles_set):
            self._cached_graphs[smiles] = \
                self.compute_graphs(smiles)
        print('\n')


    # ==========================================================================
    #                        Dataloader-related methods
    # ==========================================================================
    def train_dataloader(self) -> GeoGNNDataLoader:
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = self.shuffle,
        ) # type: ignore

    def val_dataloader(self) -> GeoGNNDataLoader:
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = self.shuffle,
        ) # type: ignore

    def test_dataloader(self) -> GeoGNNDataLoader:
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = self.shuffle,
        ) # type: ignore

    def _collate_fn(self, batch: list[GeoGNNDataElement]) -> GeoGNNBatch:
        """Collate-function used in the train/val/test dataloaders.

        Collates/Transforms a batch of `GeoGNNDataElement` obtained from the
        dataset.
        """
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        data_list: list[Tensor] = []
        for elem in batch:
            smiles, data = elem['smiles'], elem['data']
            atom_bond_graph, bond_angle_graph = self._get_graphs(smiles)
            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            data_list.append(data)
        return (
            dgl.batch(atom_bond_graphs),
            dgl.batch(bond_angle_graphs),
            self.scaler.transform(torch.stack(data_list)),
        )

    def _get_graphs(self, smiles: str) -> tuple[DGLGraph, DGLGraph]:
        """Gets cached graphs from SMILES, or compute them if they're not already cached."""
        if smiles not in self._cached_graphs:
            self._cached_graphs[smiles] \
                = self.compute_graphs(smiles)
        return self._cached_graphs[smiles]
