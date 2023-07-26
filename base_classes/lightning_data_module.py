"""Base PyTorch-Lightning data-module."""

import os, pickle
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, Literal, cast

import dgl
import lightning.pytorch as pl
import torch
from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from .dataloader import GeoGNNBatch, GeoGNNDataLoader, GeoGNNGraphs
from .dataset import GeoGNNDataElement
from .scaler import StandardizeScaler
from .transform_dataset import TransformDataset


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
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs: ...
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
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs:
        """Compute GeoGNN's atom-bond graph and bond-angle graph from a
        molecule's/reaction's SMILES/SMART string.

        Args:
            smiles (str): Molecule's/Reaction's SMILES/SMART string.

        Returns:
            GeoGNNGraphs: Atom-bond and bond-angle graphs in the \
                form `(atom_bond_graph, bond_angle_graph)`.
        """

    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
        dataloader_num_workers: int = 0,
    ) -> None:
        """
        Args:
            batch_size (int): Batch size for the dataloaders.
            shuffle (bool): Whether to shuffle training dataset. Defaults to False.
            cache_path (str | None, optional): Path to existing graph-cache file, \
                or on where to generate a new cache file should the it not exist. \
                If `None`, the graphs are computed when they're getted from the \
                dataset (instead of all being precomputed in `self.setup`). \
                Defaults to None.
            dataloader_num_workers (int, optional): Value passed to `num_workers` \
                in the train/test/val `DataLoaders`. `dataloader_num_workers > 0` \
                cannot be used with `cache_path = None`. Defaults to 0.
        """
        super().__init__()
        assert cache_path != None or dataloader_num_workers == 0, \
            "`dataloader_num_workers > 0` cannot be used with `cache_path = None` as this will " \
            + "cause CUDA to throw an error. This is because `cache_path = None` forces the graphs " \
            + "to be computed and saved to a `dict` \"cache\" when they're first getted from the " \
            + "dataset; however, I didn't design this this cache-dict to be sharable between multiple " \
            + "workers/processes. On the other hand, when `cache_path != None`, all the dataset-elements " \
            + "are transformed by the loaded/precomputed cache-dict during initialisation, thus there's " \
            + "no sharing of a cache-dict."
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_path = cache_path
        self.dataloader_num_workers = dataloader_num_workers
        self._cached_graphs: dict[str, GeoGNNGraphs] = {}

        self.raw_train_dataset, self.raw_test_dataset, self.raw_val_dataset \
            = self.get_dataset_splits()

        train_labels = torch.stack([el['data'] for el in self.raw_train_dataset])
        self.scaler = StandardizeScaler()
        """Scaler used to transform the labels for train/val/test dataloaders."""
        self.scaler.fit(train_labels)

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        self._setup_cache()
        self._preprocess_datasets()

    def _setup_cache(self) -> None:
        if not self.cache_path:
            return

        if os.path.exists(self.cache_path):
            self._load_cached_graphs()
            return

        self._precompute_all_graphs()
        self._save_cached_graphs()

    def _preprocess_datasets(self) -> None:
        preprocess_data: Callable[[GeoGNNDataElement], GeoGNNBatch] = lambda elem : (
            *self._get_graphs(elem['smiles']),
            self.scaler.transform(elem['data'])
        )
        did_not_load_cache = self.cache_path == None
        self.train_dataset = TransformDataset(self.raw_train_dataset, preprocess_data, transform_on_get=did_not_load_cache)
        self.test_dataset = TransformDataset(self.raw_test_dataset, preprocess_data, transform_on_get=did_not_load_cache)
        self.val_dataset = TransformDataset(self.raw_val_dataset, preprocess_data, transform_on_get=did_not_load_cache)

    def _get_graphs(self, smiles: str) -> GeoGNNGraphs:
        """Gets cached graphs from SMILES, or compute them if they're not already cached."""
        if smiles not in self._cached_graphs:
            self._cached_graphs[smiles] \
                = self.compute_graphs(smiles)
        return self._cached_graphs[smiles]


    # ==========================================================================
    #                          Caching-related methods
    # ==========================================================================
    def _load_cached_graphs(self) -> None:
        assert self.cache_path and os.path.exists(self.cache_path)

        # Load cached graphs dict from disk.
        print(f'Loading cached graphs file from "{self.cache_path}"...\n')
        with open(self.cache_path, 'rb') as f:
            self._cached_graphs = pickle.load(f)
        print(f'Loaded cached graphs file from "{self.cache_path}".\n')

        # Check if the SMILES in the loaded dict matches that in the full dataset.
        assert set(self._cached_graphs.keys()) == {
            data['smiles'] for data in chain(
                iter(self.raw_train_dataset),
                iter(self.raw_test_dataset),
                iter(self.raw_val_dataset),
            )
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
            data['smiles'] for data in chain(
                iter(self.raw_train_dataset),
                iter(self.raw_test_dataset),
                iter(self.raw_val_dataset),
            )
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
            num_workers = self.dataloader_num_workers,
            pin_memory = True,
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = self.shuffle,
        ) # type: ignore

    def val_dataloader(self) -> GeoGNNDataLoader:
        return DataLoader(
            num_workers = self.dataloader_num_workers,
            pin_memory = True,
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = False,
        ) # type: ignore

    def test_dataloader(self) -> GeoGNNDataLoader:
        return DataLoader(
            num_workers = self.dataloader_num_workers,
            pin_memory = True,
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            collate_fn = self._collate_fn,
            shuffle = False,
        ) # type: ignore

    @classmethod
    def _collate_fn(cls, batch: list[GeoGNNBatch]) -> GeoGNNBatch:
        """Collate-function used in the train/val/test dataloaders.

        Collates a batch of `GeoGNNBatch` obtained from the dataset.
        """
        *graph_lists, labels_list = zip(*batch)
        graph_lists = cast(list[list[DGLGraph]], graph_lists)
        labels_list = cast(list[Tensor], labels_list)

        return (
            *[dgl.batch(lst) for lst in graph_lists],
            torch.stack(labels_list),
        )
