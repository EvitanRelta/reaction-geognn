from dataclasses import dataclass
from typing import Any, Callable, Final, Literal, TypeAlias, overload

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.transforms.functional import add_reverse_edges, to_simple
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdMolTransforms as rdmt  # type: ignore
from torch import Tensor

Atom: TypeAlias = rdchem.Atom
Bond: TypeAlias = rdchem.Bond
Mol: TypeAlias = rdchem.Mol
Conformer: TypeAlias = rdchem.Conformer

FeatureCategory: TypeAlias = Literal['atom_feats', 'bond_feats']
FeatureName: TypeAlias = str

RBFFeatureCategory: TypeAlias = Literal['bond', 'bond_angle']
RBFCenters: TypeAlias = Tensor
"""1D tensor of all the RBF centers for a feature."""
RBFGamma: TypeAlias = float
"""Hyperparameter for controlling the spread of the RBF's Gaussian basis function."""


@dataclass
class Feature:
    get_value: Callable[[Atom | Bond], Any]
    """Gets the feature value from an `rdchem.Atom` / `rdchem.Bond` instance."""
    possible_values: list[Any]
    """All possible values this feature can take on."""

    def get_encoded_feat_value(self, x: Atom | Bond) -> int:
        """
        Gets the label-encoded value of the feature from `x`.

        For the feature value of `x` obtained by `self.get_value`, get the index
        of said value relative to the `self.possible_values` list, and +1 to it.

        For example:
        ```
        possible_values = ['a', 'b', 'c']
        value = get_value(x)  # if value == 'b', ie. index 1
        get_index(x)  # then `get_index` returns 1+1 = 2
        ```

        ## Note:

        The +1 is modelled after the +1 to the atom/bond ID in GeoGNN's
        preprocessing:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py#L609

        But idk what the +1 is for. It seems to be an index reserved for
        Out-Of-Vocabulary (OOV) values based on the GeoGNN comment `# 0: OOV`.
        But since `safe_index` returns the last index (which is the `"misc"`
        label of the features), unknown/unseen feature values are already mapped
        to the last index. So the "OOV" 0-th index will completely unused?

        Args:
            x (Atom | Bond): The `rdchem.Atom` / `rdchem.Bond` instance.

        Returns:
            int: Index of feature value, relative to the `self.possible_values` list.
        """
        value = self.get_value(x)
        return self.possible_values.index(value) + 1


def _to_bidirected_copy(g: DGLGraph) -> DGLGraph:
    """Exactly the same as `dgl.to_bidirected`, but copies both node and edge
    features.

    Args:
        g (DGLGraph): The input directed graph.

    Returns:
        DGLGraph: Graph `g` but bidirected and with copied node/edge features.
    """
    g = add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g = to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)      # type: ignore
    return g


class Preprocessing:
    RdChemEnum: TypeAlias = Any     # RdChem's enums have no proper typings.
    @staticmethod
    def _rdchem_enum_to_list(rdchem_enum: RdChemEnum) -> list[RdChemEnum]:
        """Converts an enum from `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType`)
        to a list of all the possible enum valuess.

        Args:
            rdchem_enum (RdChemEnum): An enum defined in `rdkit.Chem`.

        Returns:
            list[RdChemEnum]: All possible enum values in a list.
        """
        return [rdchem_enum.values[i] for i in range(len(rdchem_enum.values))]

    FEATURES: Final[dict[FeatureCategory, dict[FeatureName, Feature]]] = {
        'atom_feats': {
            'atomic_num': Feature(
                lambda atom: atom.GetAtomicNum(),
                list(range(1, 119)) + ['misc'],
            ),
            'chiral_tag': Feature(
                lambda atom: atom.GetChiralTag(),
                _rdchem_enum_to_list(rdchem.ChiralType),
            ),
            'degree': Feature(
                lambda atom: atom.GetDegree(),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            ),
            'formal_charge': Feature(
                lambda atom: atom.GetFormalCharge(),
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            ),
            'hybridization': Feature(
                lambda atom: atom.GetHybridization(),
                _rdchem_enum_to_list(rdchem.HybridizationType),
            ),
            'is_aromatic': Feature(
                lambda atom: int(atom.GetIsAromatic()),
                [0, 1],
            ),
            'total_numHs': Feature(
                lambda atom: atom.GetTotalNumHs(),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            ),
        },
        'bond_feats': {
            'bond_dir': Feature(
                lambda bond: bond.GetBondDir(),
                _rdchem_enum_to_list(rdchem.BondDir),
            ),
            'bond_type': Feature(
                lambda bond: bond.GetBondType(),
                _rdchem_enum_to_list(rdchem.BondType),
            ),
            'is_in_ring': Feature(
                lambda bond: int(bond.IsInRing()),
                [0, 1],
            ),
        }
    }
    """
    Features as defined in GeoGNN's utility functions/classes:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py

    Not all features in the above `compound_tools.py` are included, as not all
    was actually used by the GeoGNN model. Only those specified in the GeoGNN's
    config are included:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json
    """

    RBF_PARAMS: Final[dict[RBFFeatureCategory, dict[FeatureName, tuple[RBFCenters, RBFGamma]]]] = {
        'bond': {
            # Defined in GeoGNN's `BondFloatRBF` layer:
            # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L131
            'bond_length': (torch.arange(0, 2, 0.1), 10.0)
        },
        'bond_angle': {
            # Defined in GeoGNN's `BondAngleFloatRBF` layer:
            # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L168
            'bond_angle': (torch.arange(0, torch.pi, 0.1), 10.0)
        }
    }
    """
    Parameters used in RBF neural network layers, specifically the center values
    and the gamma values, where:

    - `centers` is a 1D tensor of all the RBF centers for a feature.
    - `gamma` is a hyperparameter for controlling the spread of the RBF's \
        Gaussian basis function.

    These values are defined in the constructor of `BondFloatRBF` and
    `BondAngleFloatRBF` in:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py
    """
    @overload
    @staticmethod
    def smiles_to_graphs(smiles: str, device: torch.device, return_mol_conf: Literal[True]) ->  tuple[DGLGraph, DGLGraph, Mol, Conformer]: ...
    @overload
    @staticmethod
    def smiles_to_graphs(smiles: str, device: torch.device, return_mol_conf: Literal[False] = False) -> tuple[DGLGraph, DGLGraph]: ...
    @staticmethod
    def smiles_to_graphs(
        smiles: str,
        device: torch.device = torch.device('cpu'),
        return_mol_conf: bool = False,
    ) -> tuple[DGLGraph, DGLGraph] | tuple[DGLGraph, DGLGraph, Mol, Conformer]:
        """
        Convert a molecule's SMILES string into 2 DGL graphs:
        - a graph with atoms as nodes, bonds as edges
        - a graph with bonds as nodes, bond-angles as edges

        Args:
            smiles (str): A molecule's SMILES string.
            device (torch.device): The CPU/GPU to set returned graphs to use.

        Returns:
            tuple[DGLGraph, DGLGraph]: 1st graph is the atom-bond graph, 2nd \
                is the bond-angle graph.
        """
        # Prevents `AllChem.MolFromSmiles` from removing the hydrogens explicitly
        # defined in the SMILES.
        # (but this won't add hydrogens if the SMILES doesn't have it)
        # Based on: https://github.com/rdkit/rdkit/discussions/4703#discussioncomment-1656372
        smiles_parser = Chem.SmilesParserParams()                               # type: ignore
        smiles_parser.removeHs = False
        mol = AllChem.MolFromSmiles(smiles, smiles_parser)                      # type: ignore

        mol, conf = Preprocessing._generate_conformer(mol)

        atom_bond_graph = Preprocessing._get_atom_bond_graph(mol, conf, device)
        bond_angle_graph = Preprocessing._get_bond_angle_graph(mol, conf, device)
        if return_mol_conf:
            return atom_bond_graph, bond_angle_graph, mol, conf
        return atom_bond_graph, bond_angle_graph

    @staticmethod
    def _generate_conformer(mol: Mol, numConfs: int = 10) -> tuple[Mol, Conformer]:
        try:
            new_mol = Chem.AddHs(mol)                                           # type: ignore
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)        # type: ignore
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)                    # type: ignore
            new_mol = Chem.RemoveHs(new_mol)                                    # type: ignore
            index = np.argmin([x[1] for x in res])
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)                                    # type: ignore
            conf = new_mol.GetConformer()
        return new_mol, conf

    @staticmethod
    def _get_atom_bond_graph(
        mol: Mol,
        conf: Conformer,
        device: torch.device = torch.device('cpu'),
    ) -> DGLGraph:
        """
        Gets a graph, where the nodes are the atoms in the molecule, and the
        edges are the bonds between 2 atoms.

        Args:
            mol (Mol): The `rdchem.Mol` of the molecule.
            conf (Conformer): The `rdchem.Conformer` of the molecule.
            device (torch.device): The CPU/GPU to set returned graphs to use.

        Returns:
            DGLGraph: Graph with atoms as nodes, bonds as edges.
        """
        # Create an undirected DGL graph with all the molecule's nodes and edges.
        num_bonds = mol.GetNumBonds()
        edges = torch.zeros(num_bonds, dtype=torch.int32), torch.zeros(num_bonds, dtype=torch.int32)

        has_atom_mapping = any(atom.GetAtomMapNum() != 0 for atom in mol.GetAtoms())
        if has_atom_mapping:
            for i, bond in enumerate(mol.GetBonds()):
                edges[0][i] = bond.GetBeginAtom().GetAtomMapNum() - 1
                edges[1][i] = bond.GetEndAtom().GetAtomMapNum() - 1
        else:
            for i, bond in enumerate(mol.GetBonds()):
                edges[0][i] = bond.GetBeginAtomIdx()
                edges[1][i] = bond.GetEndAtomIdx()

        graph = dgl.graph(edges, num_nodes=mol.GetNumAtoms(), idtype=torch.int32)

        # Add node features.
        for feat_name, feat in Preprocessing.FEATURES['atom_feats'].items():
            graph.ndata[feat_name] = torch.tensor([feat.get_encoded_feat_value(atom) for atom in mol.GetAtoms()])
        graph.ndata['_atom_pos'] = Preprocessing._get_atom_positions(mol, conf)

        # Add edge features.
        for feat_name, feat in Preprocessing.FEATURES['bond_feats'].items():
            graph.edata[feat_name] = torch.tensor([feat.get_encoded_feat_value(bond) for bond in mol.GetBonds()])
        graph.edata['bond_length'] = Preprocessing._get_bond_lengths(graph)

        # Remove temporary feats used in computing other feats.
        del graph.ndata['_atom_pos']

        graph = _to_bidirected_copy(graph)   # Convert to undirected graph.
        graph = graph.to(device)    # Move graph to CPU/GPU depending on `device`.
        return graph

    @staticmethod
    def _get_bond_angle_graph(
        mol: Mol,
        conf: Conformer,
        device: torch.device = torch.device('cpu'),
    ) -> DGLGraph:
        """
        Gets a graph, where the nodes are the bonds in the molecule, and the
        edges are the angles between 2 bonds.

        Args:
            mol (Mol): The `rdchem.Mol` of the molecule.
            conf (Conformer): The `rdchem.Conformer` of the molecule.
            device (torch.device): The CPU/GPU to set returned graphs to use.

        Returns:
            DGLGraph: Graph with bonds as nodes, bond-angles as edges.
        """
        # Number of bonds in the molecule.
        num_of_bonds = mol.GetNumBonds()

        # Initialize graph with 1 node per bond.
        graph = dgl.graph(([], []), num_nodes=num_of_bonds)

        # For the edge case where there's no bonds.
        if num_of_bonds == 0:
            graph.edata['bond_angle'] = torch.tensor([])
            graph = graph.to(device)    # Move graph to CPU/GPU depending on `device`.
            return graph

        # Calculate and store bond angles for each pair of bonds that share an atom.
        for i in range(num_of_bonds):
            bond_i = mol.GetBondWithIdx(i)
            for j in range(i+1, num_of_bonds):
                bond_j = mol.GetBondWithIdx(j)

                # Get the 4 atoms of the 2 bonds.
                atom_indexes: list[int] = [
                    bond_i.GetBeginAtomIdx(),
                    bond_i.GetEndAtomIdx(),
                    bond_j.GetBeginAtomIdx(),
                    bond_j.GetEndAtomIdx(),
                ]
                unique_atom_indexes = list(set(atom_indexes))

                # Check if bonds share an atom.
                if len(unique_atom_indexes) == 3:
                    shared_atom_index = [x for x in atom_indexes if atom_indexes.count(x) > 1][0]
                    ends_indexes = [x for x in atom_indexes if atom_indexes.count(x) == 1]

                    # If so, calculate the bond angle.
                    angle = rdmt.GetAngleRad(conf, ends_indexes[0], shared_atom_index, ends_indexes[1])

                    # Add an edge to the graph for this bond angle.
                    graph.add_edges(i, j, {'bond_angle': torch.tensor([angle])})

        graph = _to_bidirected_copy(graph)   # Convert to undirected graph.
        graph = graph.to(device)    # Move graph to CPU/GPU depending on `device`.
        return graph

    @staticmethod
    def _get_atom_positions(mol: Mol, conf: Conformer) -> Tensor:
        """
        Gets the 3D-coords of all atoms.

        Args:
            mol (Mol): The `rdchem.Mol` of the molecule.
            conf (Conformer): The `rdchem.Conformer` of the molecule.

        Returns:
            Tensor: 3D-coords of all atoms, shape `(num_of_atoms, 3)`, dtype=float32.
        """
        # Convert to float32 Tensor, from float64 Numpy array.
        raw_atom_positions = torch.from_numpy(conf.GetPositions()).float()

        # Truncate tensor as `mol` likely won't have Hydrogen atoms,
        # while `conf` likely will. Since the H atoms are placed after the
        # non-H atoms in the tensor, the H atoms positions will be truncated.
        return raw_atom_positions[:mol.GetNumAtoms()]

    @staticmethod
    def _get_bond_lengths(graph: DGLGraph) -> Tensor:
        """
        Gets all the bond lengths in a molecule.

        Note: This requires the 3D-coords of the atoms to already be computed at
        `graph.ndata['_atom_pos']`.

        Args:
            graph (DGLGraph): The graph with atoms as nodes, bonds as edges, and \
                with the 3D-coords of the atoms already computed at \
                `graph.ndata['_atom_pos']`.

        Returns:
            Tensor: Bond lengths of all the bonds in the molecule.
        """
        atom_positions: Tensor = graph.ndata['_atom_pos']                       # type: ignore
        edges_tuple: tuple[Tensor, Tensor] = graph.edges()

        src_node_idx, dst_node_idx = edges_tuple

        # To use as tensor indexes, these index tensors needs to be dtype=long.
        src_node_idx, dst_node_idx = src_node_idx.long(), dst_node_idx.long()

        return torch.norm(atom_positions[dst_node_idx] - atom_positions[src_node_idx], dim=1)
