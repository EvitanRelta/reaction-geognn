from abc import ABC
from collections.abc import Iterator
from typing import TypeAlias

from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import GeoGNNDataElement

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
