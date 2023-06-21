from numbers import Number
from typing import Any

import torch
from dgl import DGLGraph
from torch import FloatTensor, IntTensor, Tensor

from .rdkit_types import Atom, Bond, Conformer, Mol


class Feature:
    def __init__(self, name: str) -> None:
        self.name = name
        """Name of this feature, which will be the feature's key in `DGLGraph.ndata` / `.edata`."""

    # Methods that needs implementing.
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor: raise NotImplementedError()


class FloatFeature(Feature):
    # Variables/Methods that needs implementing.
    centers: list[float]
    gamma: float
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> FloatTensor: raise NotImplementedError()


class LabelEncodedFeature(Feature):
    # Variables/Methods that needs implementing.
    possible_values: list[Any]
    def _get_unencoded_value(self, x: Atom | Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Any: raise NotImplementedError()
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> IntTensor: raise NotImplementedError()

    # Implementations.
    def _get_value(self, x: Atom | Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> int:
        value = self._get_unencoded_value(x, mol, conf, atom_bond_graph)
        try:
            return self.possible_values.index(value) + 1
        except ValueError:
            # If `value` doesn't exist in `self.possible_values`,
            # return the last element's index (assumes 'misc' is the last index).
            return len(self.possible_values) - 1


class AtomFeature(Feature):
    # Variables/Methods that needs implementing.
    def _get_value(self, x: Atom, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Number: raise NotImplementedError()

    # Implementations.
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor:
        return torch.tensor([
            self._get_value(atom, mol, conf, atom_bond_graph) \
                for atom in mol.GetAtoms()
        ])


class BondFeature(Feature):
    # Variables/Methods that needs implementing.
    def _get_value(self, x: Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Number: raise NotImplementedError()

    # Implementations.
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor:
        return torch.tensor([
            self._get_value(bond, mol, conf, atom_bond_graph) \
                for bond in mol.GetBonds()
        ])
