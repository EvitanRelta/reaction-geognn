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
from dgl import DGLGraph
from rdkit.Chem import rdchem  # type: ignore
from torch import Tensor

from .base_feature_classes import Feature, FloatFeature, LabelEncodedFeature
from .rdkit_types import Conformer, Mol, RDKitEnum, RDKitEnumValue


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

atomic_num_atom_feat = LabelEncodedFeature(
    name = 'atomic_num',
    feat_type = 'atom',
    possible_values = list(range(1, 119)) + ['misc'],
    get_raw_value = lambda x, *_ : x.GetAtomicNum(),
)


chiral_tag_atom_feat = LabelEncodedFeature(
    name = 'chiral_tag',
    feat_type = 'atom',
    possible_values = _rdkit_enum_to_list(rdchem.ChiralType),
    get_raw_value = lambda x, *_ : x.GetChiralTag(),
)


degree_atom_feat = LabelEncodedFeature(
    name = 'degree',
    feat_type = 'atom',
    possible_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    get_raw_value = lambda x, *_ : x.GetDegree(),
)


formal_charge_atom_feat = LabelEncodedFeature(
    name = 'formal_charge',
    feat_type = 'atom',
    possible_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    get_raw_value = lambda x, *_ : x.GetFormalCharge(),
)


hybridization_atom_feat = LabelEncodedFeature(
    name = 'hybridization',
    feat_type = 'atom',
    possible_values = _rdkit_enum_to_list(rdchem.HybridizationType),
    get_raw_value = lambda x, *_ : x.GetHybridization(),
)


is_aromatic_atom_feat = LabelEncodedFeature(
    name = 'is_aromatic',
    feat_type = 'atom',
    possible_values = [0, 1],
    get_raw_value = lambda x, *_ : x.GetIsAromatic(),
    dtype = torch.bool,
)


total_numHs_atom_feat = LabelEncodedFeature(
    name = 'total_numHs',
    feat_type = 'atom',
    possible_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    get_raw_value = lambda x, *_ : x.GetTotalNumHs(),
)


atom_pos_atom_feat = Feature(
    name = '_atom_pos',
    get_feat_values = lambda _, conf, *__ : torch.from_numpy(conf.GetPositions()).float(),
)


LABEL_ENCODED_ATOM_FEATURES: Final[list[LabelEncodedFeature]] = [
    atomic_num_atom_feat,
    chiral_tag_atom_feat,
    degree_atom_feat,
    formal_charge_atom_feat,
    hybridization_atom_feat,
    is_aromatic_atom_feat,
    total_numHs_atom_feat,
]
"""
All predefined label-encoded atom features that'll be in the graphs.
"""

FLOAT_ATOM_FEATURES: Final[list[Feature]] = []
"""
All predefined atom features that have feature values of datatype `float`
that'll be in the graphs.
"""

# ==============================================================================
#                                  Bond features
# ==============================================================================
bond_dir_bond_feat = LabelEncodedFeature(
    name = 'bond_dir',
    feat_type = 'bond',
    possible_values = _rdkit_enum_to_list(rdchem.BondDir),
    get_raw_value = lambda x, *_ : x.GetBondDir(),
)


bond_type_bond_feat = LabelEncodedFeature(
    name = 'bond_type',
    feat_type = 'bond',
    possible_values = _rdkit_enum_to_list(rdchem.BondType),
    get_raw_value = lambda x, *_ : x.GetBondType(),
)


is_in_ring_bond_feat = LabelEncodedFeature(
    name = 'is_in_ring',
    feat_type = 'bond',
    possible_values = [0, 1],
    get_raw_value = lambda x, *_ : int(x.IsInRing()),
    dtype = torch.bool,
)


def _bond_length_get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None):
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

bond_length_bond_feat = FloatFeature(
    name = 'bond_length',
    centers = torch.arange(0, 2, 0.1),
    gamma = 10.0,
    get_feat_values = _bond_length_get_feat_values,
)


LABEL_ENCODED_BOND_FEATURES: Final[list[LabelEncodedFeature]] = [
    bond_dir_bond_feat,
    bond_type_bond_feat,
    is_in_ring_bond_feat,
]
"""
All predefined label-encoded bond features that'll be in the graphs.
"""

FLOAT_BOND_FEATURES: Final[list[Feature]] = [
    bond_length_bond_feat,
]
"""
All predefined bond features that have feature values of datatype `float`
that'll be in the graphs.
"""
