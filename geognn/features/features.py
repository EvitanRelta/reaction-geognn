"""
Features as defined in GeoGNN's utility functions/classes:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py

Not all features in the above `compound_tools.py` are included, as not all
was actually used by the GeoGNN model. Only those specified in the GeoGNN's
config are included:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json
"""

from typing import Final

import torch
from rdkit.Chem import rdchem  # type: ignore
from torch import Tensor
from typing_extensions import override

from .base_feature_classes import AtomFeature, BondFeature, Feature, \
    LabelEncodedFeature
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
class AtomicNum(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return list(range(1, 119)) + ['misc']
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetAtomicNum()


class ChiralTag(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.ChiralType)
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetChiralTag()


class Degree(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetDegree()


class FormalCharge(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetFormalCharge()


class Hybridization(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.HybridizationType)
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetHybridization()


class IsAromatic(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return [0, 1]
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return int(x.GetIsAromatic())


class TotalNumHs(AtomFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return x.GetTotalNumHs()


class AtomPosition(AtomFeature):
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        raise NotImplementedError("Atom positions are not computed atom-by-atom, but all at once in `get_feat_values`.")
    @override
    def get_feat_values(self, mol, conf, atom_bond_graph = None) -> Tensor:
        return torch.from_numpy(conf.GetPositions()).float()


ATOM_FEATURES: Final[list[Feature]] = [
    AtomicNum('atomic_num'),
    ChiralTag('chiral_tag'),
    Degree('degree'),
    FormalCharge('formal_charge'),
    Hybridization('hybridization'),
    IsAromatic('is_aromatic'),
    TotalNumHs('total_numHs'),
    AtomPosition('_atom_pos'),  # Temp feat used for generating bond lengths.
]
"""
All predefined atom features.
"""


# ==============================================================================
#                                  Bond features
# ==============================================================================
class BondDirection(BondFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.BondDir)
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetBondDir()


class BondType(BondFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return _rdkit_enum_to_list(rdchem.BondType)
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> RDKitEnumValue:
        return x.GetBondType()


class IsInRing(BondFeature, LabelEncodedFeature):
    @override
    def _get_possible_values(self):
        return [0, 1]
    @override
    def _get_unencoded_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        return int(x.IsInRing())


class BondLength(BondFeature):
    @override
    def _get_value(self, x, mol, conf, atom_bond_graph = None) -> int:
        raise NotImplementedError("Bond lengths are not computed bond-by-bond, but all at once in `get_feat_values`.")
    @override
    def get_feat_values(self, mol, conf, atom_bond_graph = None) -> Tensor:
        assert atom_bond_graph != None, 'Bond length feature requires `atom_bond_graph`.'
        assert '_atom_pos' in atom_bond_graph.ndata, \
            "Bond length feat requires 3D atom position feat to be first computed at `atom_bond_graph.ndata['_atom_pos']`."

        atom_positions = atom_bond_graph.ndata['_atom_pos']
        assert isinstance(atom_positions, Tensor)

        edges_tuple: tuple[Tensor, Tensor] = atom_bond_graph.edges()
        src_node_idx, dst_node_idx = edges_tuple

        # To use as tensor indexing, these index tensors needs to be `dtype=long`.
        src_node_idx, dst_node_idx = src_node_idx.long(), dst_node_idx.long()

        return torch.norm(atom_positions[dst_node_idx] - atom_positions[src_node_idx], dim=1)


BOND_FEATURES: Final[list[Feature]] = [
    BondDirection('bond_dir'),
    BondType('bond_type'),
    IsInRing('is_in_ring'),
    BondLength('bond_length'),
]
"""
All predefined bond features.
"""
