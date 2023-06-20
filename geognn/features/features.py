from typing import Final

import torch
from rdkit.Chem import rdchem  # type: ignore
from torch import Tensor
from typing_extensions import override

from .base_feature_classes import AtomFeature, BondFeature
from .rdkit_types import RDKitEnum, RDKitEnumValue


def _rdkit_enum_to_list(rdkit_enum: RDKitEnum) -> list[RDKitEnumValue]:
    """
    Converts an enum from `rdkit.Chem.rdchem` (eg. `rdkit.Chem.rdchem.ChiralType`)
    to a list of all the possible enum values
    (eg. `[ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ...]`)

    Args:
        rdkit_enum (RDKitEnum): An enum from `rdkit.Chem.rdchem`.

    Returns:
        list[RDKitEnumValue]: All possible enum values in a list.
    """
    return [rdkit_enum.values[i] for i in range(len(rdkit_enum.values))]


# ==============================================================================
#                                  Atom features
# ==============================================================================
class AtomicNum(AtomFeature):
    @override
    def _get_possible_values(self):
        return list(range(1, 119)) + ['misc']
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetAtomicNum()


class ChiralTag(AtomFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.ChiralType)
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetChiralTag()


class Degree(AtomFeature):
    @override
    def _get_possible_values(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetDegree()


class FormalCharge(AtomFeature):
    @override
    def _get_possible_values(self):
        return [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetFormalCharge()


class Hybridization(AtomFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.HybridizationType)
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetHybridization()


class IsAromatic(AtomFeature):
    @override
    def _get_possible_values(self):
        return [0, 1]
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return int(x.GetIsAromatic())


class TotalNumHs(AtomFeature):
    @override
    def _get_possible_values(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetTotalNumHs()


class AtomPosition(AtomFeature):
    @override
    def _get_possible_values(self):
        raise NotImplementedError("Atom positions doesn't have finite number of possible values.")
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        raise NotImplementedError("Atom positions are not computed atom-by-atom, but all at once in `get_feat_values`.")
    @override
    def get_feat_values(self, mol, conf, atom_bond_graph = None) -> Tensor:
        return torch.from_numpy(conf.GetPositions()).float()


ATOM_FEATURES: Final[list[AtomFeature]] = [
    AtomicNum('atomic_num'),
    ChiralTag('chiral_tag'),
    Degree('degree'),
    FormalCharge('formal_charge'),
    Hybridization('hybridization'),
    IsAromatic('is_aromatic'),
    TotalNumHs('total_numHs'),
    AtomPosition('_atom_pos'),  # Temp feat used for generating bond lengths.
]


# ==============================================================================
#                                  Bond features
# ==============================================================================
class BondDirection(BondFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.BondDir)
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetBondDir()


class BondType(BondFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.BondType)
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetBondType()


class IsInRing(BondFeature):
    @override
    def _get_possible_values(self):
        return [0, 1]
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return int(x.IsInRing())


BOND_FEATURES: Final[list[BondFeature]] = [
    BondDirection('bond_dir'),
    BondType('bond_type'),
    IsInRing('is_in_ring'),
]
