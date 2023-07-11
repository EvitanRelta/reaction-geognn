import os, pickle
from typing import Literal

import dgl
import lightning.pytorch as pl
import torch
from dgl import DGLGraph
from geognn.datasets import GeoGNNDataElement
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .datasets import get_wb97_fold_dataset
from .preprocessing import reaction_smart_to_graph

BATCH_TUPLE = tuple[DGLGraph, DGLGraph, Tensor]
"""Batched input in the form `(atom_bond_batch_graph, bond_angle_batch_graph, labels)`"""

class Wb97DataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold_num: Literal[0, 1, 2, 3, 4],
        batch_size: int | None,
        cache_path: str | None = None
    ):
        super().__init__()
        self.fold_num: Literal[0, 1, 2, 3, 4] = fold_num
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {}

        self.train_dataset, self.test_dataset, self.val_dataset \
            = get_wb97_fold_dataset(self.fold_num)

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
                self.train_dataset.data_list \
                + self.test_dataset.data_list \
                + self.val_dataset.data_list
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
                self.train_dataset.data_list \
                + self.test_dataset.data_list \
                + self.val_dataset.data_list
        }
        print(f'Precomputing graphs for {len(full_smiles_set)} SMILES strings:')
        for smiles in tqdm(full_smiles_set):
            self._cached_graphs[smiles] = \
                reaction_smart_to_graph(smiles, device=torch.device('cpu'))
        print('\n')


    # ==========================================================================
    #                        Dataloader-related methods
    # ==========================================================================
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
        )

    def _collate_fn(self, batch: list[GeoGNNDataElement]) -> BATCH_TUPLE:
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
                = reaction_smart_to_graph(smiles, torch.device('cpu'))
        return self._cached_graphs[smiles]
