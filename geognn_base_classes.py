from abc import ABC
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class GeoGNNDataElement(TypedDict):
    """A data entry for GeoGNN."""

    smiles: str
    """SMILES string of the data's molecule."""

    data: Tensor
    """Ground truth data. Size `(num_of_feats, num_of_entries)`"""


@dataclass
class GeoGNNDataset(ABC, Dataset[GeoGNNDataElement], Sized):
    """
    Abstract base class for a dataset used by GeoGNN.

    Constructor takes in a `list[GeoGNNDataElement]` which will be accessible
    via `self.data_list`.

    This class implements `__getitem__` and `__len__` which returns the elements
    of and the length of `self.data_list` respectively.

    Data elements are dictionaries of type `GeoGNNDataElement`, each containing
    the keys `"smiles"` (the element's SMILES string) and `"data"` (tensor of
    the element's data/labels).
    """

    data_list: list[GeoGNNDataElement]
    """List of the data elements. The data elements are dictionaries of type
    `GeoGNNDataElement`, each containing the keys `"smiles"` (the element's
    SMILES string) and `"data"` (tensor of the element's data/labels)."""

    def __getitem__(self, index: int) -> GeoGNNDataElement:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)


GeoGNNBatch: TypeAlias = tuple[DGLGraph, DGLGraph, Tensor]
"""Batched input in the form `(atom_bond_batch_graph, bond_angle_batch_graph, labels)`
of type `tuple[DGLGraph, DGLGraph, Tensor]`.
"""

class GeoGNNDataLoader(ABC, DataLoader[GeoGNNDataElement]):
    """
    Abstract base class for a data-loader used by GeoGNN.

    Outputs batches of type `GeoGNNBatch`, which is in the form
    `(atom_bond_batch_graph, bond_angle_batch_graph, labels)` of type
    `tuple[DGLGraph, DGLGraph, Tensor]`.

    Expects a dataset of type `Dataset[GeoGNNDataElement]`, where the data
    elements are dictionaries of type `GeoGNNDataElement`, each containing
    the keys `"smiles"` (the element's SMILES string) and `"data"` (tensor of
    the element's data/labels).
    """
    def __iter__(self) -> Iterator[GeoGNNBatch]:
        return super().__iter__()
