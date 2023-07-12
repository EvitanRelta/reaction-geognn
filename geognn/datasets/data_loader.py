from typing import Callable, Iterator

import dgl, torch
from dgl import DGLGraph
from geognn_base_classes import GeoGNNDataElement, GeoGNNDataLoader
from torch import Tensor
from torch.utils.data import Dataset

from .. import Preprocessing


class GeoGNNDataLoader_(GeoGNNDataLoader):
    """
    Data loader for GeoGNN's datasets.
    """

    def __init__(
        self,
        dataset: Dataset[GeoGNNDataElement],
        fit_mean: Tensor,
        fit_std: Tensor,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu'),
        cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {},
        collate_fn: Callable | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn = self._collate_fn if collate_fn is None \
                else collate_fn,
            worker_init_fn = worker_init_fn,
            generator = generator,
        )
        self.device = device
        self.fit_mean = fit_mean.to(device)
        self.fit_std = fit_std.to(device)
        self._cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = cached_graphs

    @staticmethod
    def get_stats(dataset: Dataset[GeoGNNDataElement]) -> tuple[Tensor, Tensor]:
        """
        Gets the mean and standard deviation of a GeoGNN dataset.

        Args:
            dataset (Dataset[GeoGNNDataElement]): Dataset containing \
                `GeoGNNDataElement` elements, and implements `__len__`.

        Returns:
            tuple[Tensor, Tensor]: The mean & standard deviation in the form - \
                `(mean, std)`.
        """
        # Extract the data as a `Tensor``.
        data_list = [dataset[i]['data'] for i in range(len(dataset))]   # type: ignore
        device = data_list[0].device
        data = torch.stack(data_list).to(device)

        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return mean, std

    def _collate_fn(self, batch: list[GeoGNNDataElement]) -> tuple[DGLGraph, DGLGraph, Tensor]:
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        data_list: list[Tensor] = []
        for elem in batch:
            smiles, data = elem['smiles'], elem['data']

            if smiles in self._cached_graphs:
                atom_bond_graph, bond_angle_graph = self._cached_graphs[smiles]
            else:
                graphs = Preprocessing.smiles_to_graphs(smiles, self.device)
                atom_bond_graph, bond_angle_graph = graphs
                self._cached_graphs[smiles] = graphs

            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            data_list.append(data)

        data = torch.stack(data_list).to(self.device)
        return (
            dgl.batch(atom_bond_graphs),
            dgl.batch(bond_angle_graphs),
            GeoGNNDataLoader_._standardize_data(data, self.fit_mean, self.fit_std),
        )

    @staticmethod
    def _standardize_data(
        data: Tensor,
        fit_mean: Tensor,
        fit_std: Tensor,
        epsilon: float = 1e-5,
    ) -> Tensor:
        """
        Standardize each feature column by the training data's mean and standard
        deviation - `fit_mean` and `fit_std` respectively.

        Args:
            data (Tensor): The data, where each column is a feature.
            fit_mean: (float): The mean to based the standardization on \
                (eg. the training data's mean).
            fit_std: (float): The standard deviation to based the \
                standardization on (eg. the training data's standard deviation).
            epsilon (float, optional): Small number to avoid division-by-zero \
                errors. Defaults to 1e-5.
        """
        return (data - fit_mean) / (fit_std + epsilon)

    def unstandardize_data(
        self,
        standardized_data: Tensor,
    ) -> Tensor:
        """
        Reverse the standardization on the feature columns using the
        `self.fit_mean` and `self.fit_std` (ie. the training data's mean and
        standard deviation respectively).

        Args:
            standardized_data (Tensor): The standardized data, where each column \
                is a feature that has been standardized.
            fit_mean: (float): The mean to based the standardization on \
                (eg. the training data's mean).
            fit_std: (float): The standard deviation to based the \
                standardization on (eg. the training data's standard deviation).
            epsilon (float, optional): Small number to avoid division-by-zero \
                errors. Defaults to 1e-5.
        """
        return (standardized_data * self.fit_std) + self.fit_mean

    def __iter__(self) -> Iterator[tuple[DGLGraph, DGLGraph, Tensor]]:
        return super().__iter__()
