from abc import ABC
from collections.abc import Sized
from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor
from torch.utils.data import Dataset


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
