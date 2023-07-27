from dataclasses import dataclass
from typing import Any, Callable

import torch
from dgl import DGLGraph
from torch import FloatTensor, IntTensor, Tensor

from .rdkit_type_aliases import Atom, Bond, Conformer, Mol


@dataclass
class Feature:
    name: str
    get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], Tensor]

    @staticmethod
    def create_atom_feat(
        name: str,
        get_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], int | float],
        dtype: torch.dtype,
    ) -> 'Feature':
        return Feature(
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([
                get_value(atom, mol, conf, atom_bond_graph) for atom in mol.GetAtoms()
            ], dtype=dtype),
        )

    @staticmethod
    def create_bond_feat(
        name: str,
        get_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], int | float],
        dtype: torch.dtype
    ) -> 'Feature':
        return Feature(
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([
                get_value(bond, mol, conf, atom_bond_graph) for bond in mol.GetBonds()
            ], dtype=dtype),
        )


@dataclass
class FloatFeature(Feature):
    # Overridden fields from `Feature` class.
    name: str
    get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], FloatTensor]

    rbf_centers: Tensor
    rbf_gamma: float

    @staticmethod
    def create_atom_feat(
        name: str,
        rbf_centers: Tensor,
        rbf_gamma: float,
        get_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], float],
        dtype: torch.dtype = torch.float32,
    ) -> 'FloatFeature':
        return FloatFeature(
            rbf_centers = rbf_centers,
            rbf_gamma = rbf_gamma,
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                get_value(atom, mol, conf, atom_bond_graph) for atom in mol.GetAtoms()
            ], dtype=dtype),
        )

    @staticmethod
    def create_bond_feat(
        name: str,
        rbf_centers: Tensor,
        rbf_gamma: float,
        get_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], float],
        dtype: torch.dtype = torch.float32,
    ) -> 'FloatFeature':
        return FloatFeature(
            rbf_centers = rbf_centers,
            rbf_gamma = rbf_gamma,
            name = name,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                get_value(bond, mol, conf, atom_bond_graph) for bond in mol.GetBonds()
            ], dtype=dtype),
        )


@dataclass
class LabelEncodedFeature(Feature):
    # Overridden fields from `Feature` class.
    name: str
    get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], IntTensor]

    possible_values: list[Any]

    @staticmethod
    def create_atom_feat(
        name: str,
        possible_values: list[Any],
        get_raw_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], Any],
        dtype: torch.dtype = torch.uint8,
    ) -> 'LabelEncodedFeature':
        return LabelEncodedFeature(
            name = name,
            possible_values = possible_values,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                LabelEncodedFeature._encoded_value(
                    raw_value = get_raw_value(atom, mol, conf, atom_bond_graph),
                    possible_values = possible_values,
                ) for atom in mol.GetAtoms()
            ], dtype=dtype),
        )

    @staticmethod
    def create_bond_feat(
        name: str,
        possible_values: list[Any],
        get_raw_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], Any],
        dtype: torch.dtype = torch.uint8,
    ) -> 'LabelEncodedFeature':
        return LabelEncodedFeature(
            name = name,
            possible_values = possible_values,
            get_feat_values = lambda mol, conf, atom_bond_graph : torch.tensor([ # type: ignore
                LabelEncodedFeature._encoded_value(
                    raw_value = get_raw_value(bond, mol, conf, atom_bond_graph),
                    possible_values = possible_values,
                ) for bond in mol.GetBonds()
            ], dtype=dtype),
        )

    @staticmethod
    def _encoded_value(
        raw_value: Any,
        possible_values: list[Any],
    ) -> int:
        try:
            return possible_values.index(raw_value) + 1
        except ValueError:
            # If `value` doesn't exist in `self.possible_values`,
            # return the last element's index (assumes 'misc' is the last index).
            return len(possible_values) - 1
