from abc import ABC
from collections.abc import Iterator
from typing import TypeAlias

from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Unpack

from .dataset import GeoGNNDataElement

GeoGNNGraphs: TypeAlias = tuple[DGLGraph, ...]
"""The graphs in `GeoGNNBatch`. Can be either be `(atom_bond_graph, bond_angle_graph)`
or `(atom_bond_graph, bond_angle_graph, superimposed_atom_bond_graph)` depending
on the dataset. The graphs are of type `DGLGraph`.
"""

GeoGNNBatch: TypeAlias = tuple[Unpack[GeoGNNGraphs], Tensor]
"""Batched (or unbatched, since both are of the same type) data in the form
`(batched_atom_bond_graph, batched_bond_angle_graph, labels)`, or
`(batched_atom_bond_graph, batched_bond_angle_graph, batched_superimposed_atom_bond_graph, labels)`
depending on the dataset.

The graphs are of type `DGLGraph`, while `labels` is of type `Tensor`.
"""


class GeoGNNDataLoader(ABC, DataLoader[GeoGNNDataElement]):
    """Abstract base class for a data-loader used by GeoGNN.

    Outputs batches of type `GeoGNNBatch`, which is in the form
    `(batched_atom_bond_graph, batched_bond_angle_graph, labels)` or
    `(batched_atom_bond_graph, batched_bond_angle_graph, batched_superimposed_atom_bond_graph, labels)`,
    where the graphs are of type `DGLGraph`, while `labels` is of type `Tensor`.

    Expects a dataset of type `Dataset[GeoGNNDataElement]`, where the data
    elements are dictionaries of type `GeoGNNDataElement`, each containing
    the keys `"smiles"` (the element's SMILES string) and `"data"` (tensor of
    the element's data/labels).
    """
    def __iter__(self) -> Iterator[GeoGNNBatch]:
        return super().__iter__()
