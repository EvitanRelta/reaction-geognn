from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Final, Generic, TypeVar

import torch
from dgl import DGLGraph
from torch import Tensor
from typing_extensions import override

from .rdkit_types import Atom, Bond, Conformer, Mol

T = TypeVar('T', Atom, Bond)
"""The types that the `Feature` class can take in."""

class Feature(ABC, Generic[T]):
    def __init__(self, name: str):
        self._possible_values: list[Any] | None = None
        self.name: str = name

    @property
    def possible_values(self) -> list[Any]:
        if self._possible_values == None:
            self._possible_values = self._get_possible_values()
        return self._possible_values

    @abstractmethod
    def _get_possible_values(self) -> list[Any]: ...

    @abstractmethod
    def _get_value(self, x: T, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Any: ...

    def _get_encoded_value(self, x: T, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> int:
        value = self._get_value(x, mol, conf, atom_bond_graph)
        try:
            return self.possible_values.index(value) + 1
        except ValueError:
            # If `value` doesn't exist in `self.possible_values`,
            # return the last element's index (assumes 'misc' is the last index).
            return len(self.possible_values) - 1

    @abstractmethod
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor: ...

class AtomFeature(Feature[Atom]):
    @override
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor:
        return torch.tensor([
            self._get_encoded_value(atom, mol, conf, atom_bond_graph) \
                for atom in mol.GetAtoms()
        ])

class BondFeature(Feature[Bond]):
    @override
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor:
        return torch.tensor([
            self._get_encoded_value(bond, mol, conf, atom_bond_graph) \
                for bond in mol.GetBonds()
        ])
