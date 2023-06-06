import dgl, torch, numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, rdMolTransforms as rdmt
from dgl import DGLGraph
from dgl.transforms.functional import add_reverse_edges, to_simple
from typing import TypeAlias, Any, Callable, Literal, Final
from dataclasses import dataclass
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


def to_bidirected_copy(g: DGLGraph) -> DGLGraph:
    """Exactly the same as `dgl.to_bidirected`, but copies both node and edge
    features.

    Args:
        g (DGLGraph): The input directed graph.

    Returns:
        DGLGraph: Graph `g` but bidirected and with copied node/edge features.
    """
    g = add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g = to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)
    return g

class Utils:
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

    @staticmethod
    def smiles_to_graphs(smiles: str, use_gpu: bool = True) -> tuple[DGLGraph, DGLGraph]:
        """
        Convert a molecule's SMILES string into 2 DGL graphs:
        - a graph with atoms as nodes, bonds as edges
        - a graph with bonds as nodes, bond-angles as edges

        Args:
            smiles (str): A molecule's SMILES string.
            use_gpu (bool): If `True`, set returned graphs to use GPU, else use CPU.

        Returns:
            tuple[DGLGraph, DGLGraph]: 1st graph is the atom-bond graph, 2nd \
                is the bond-angle graph.
        """
        mol = AllChem.MolFromSmiles(smiles)
        mol, conf = Utils._generate_conformer(mol)

        atom_bond_graph = Utils._get_atom_bond_graph(mol, conf, use_gpu)
        bond_angle_graph = Utils._get_bond_angle_graph(mol, conf, use_gpu)
        return atom_bond_graph, bond_angle_graph

    @staticmethod
    def _generate_conformer(mol: Mol, numConfs: int = 10) -> tuple[Mol, Conformer]:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        conf = new_mol.GetConformer(id=int(index))
        return new_mol, conf

    @staticmethod
    def _get_atom_bond_graph(mol: Mol, conf: Conformer, use_gpu: bool = True) -> DGLGraph:
        """
        Gets a graph, where the nodes are the atoms in the molecule, and the
        edges are the bonds between 2 atoms.

        Args:
            mol (Mol): The `rdchem.Mol` of the molecule.
            conf (Conformer): The `rdchem.Conformer` of the molecule.
            use_gpu (bool): If `True`, set returned graph to use GPU, else use CPU.

        Returns:
            DGLGraph: Graph with atoms as nodes, bonds as edges.
        """
        # Create an undirected DGL graph with all the molecule's nodes and edges.
        num_bonds = mol.GetNumBonds()
        edges = torch.zeros(num_bonds, dtype=torch.int32), torch.zeros(num_bonds, dtype=torch.int32)
        for i, bond in enumerate(mol.GetBonds()):
            edges[0][i] = bond.GetBeginAtomIdx()
            edges[1][i] = bond.GetEndAtomIdx()
        graph = dgl.graph(edges, idtype=torch.int32)

        # Add node features.
        for feat_name, feat in Utils.FEATURES['atom_feats'].items():
            graph.ndata[feat_name] = torch.tensor([feat.get_encoded_feat_value(atom) for atom in mol.GetAtoms()])
        graph.ndata['atom_pos'] = Utils._get_atom_positions(mol, conf)

        # Add edge features.
        for feat_name, feat in Utils.FEATURES['bond_feats'].items():
            graph.edata[feat_name] = torch.tensor([feat.get_encoded_feat_value(bond) for bond in mol.GetBonds()])
        graph.edata['bond_length'] = Utils._get_bond_lengths(graph)

        graph = to_bidirected_copy(graph)   # Convert to undirected graph.
        if use_gpu:
            # Copies graph to GPU. (https://docs.dgl.ai/guide/graph-gpu.html)
            graph = graph.to('cuda:0')
        return graph

    @staticmethod
    def _get_bond_angle_graph(mol: Mol, conf: Conformer, use_gpu: bool = True) -> DGLGraph:
        """
        Gets a graph, where the nodes are the bonds in the molecule, and the
        edges are the angles between 2 bonds.

        Args:
            mol (Mol): The `rdchem.Mol` of the molecule.
            conf (Conformer): The `rdchem.Conformer` of the molecule.
            use_gpu (bool): If `True`, set returned graph to use GPU, else use CPU.

        Returns:
            DGLGraph: Graph with bonds as nodes, bond-angles as edges.
        """
        # Number of bonds in the molecule.
        num_of_bonds = mol.GetNumBonds()

        # Initialize graph with 1 node per bond.
        graph = DGLGraph()
        graph.add_nodes(num_of_bonds)

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

        graph = to_bidirected_copy(graph)   # Convert to undirected graph.
        if use_gpu:
            # Copies graph to GPU. (https://docs.dgl.ai/guide/graph-gpu.html)
            graph = graph.to('cuda:0')
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
        `graph.ndata['atom_pos']`.

        Args:
            graph (DGLGraph): The graph with atoms as nodes, bonds as edges, and \
                with the 3D-coords of the atoms already computed at \
                `graph.ndata['atom_pos']`.

        Returns:
            Tensor: Bond lengths of all the bonds in the molecule.
        """
        atom_positions: Tensor = graph.ndata['atom_pos']
        edges_tuple: tuple[Tensor, Tensor] = graph.edges()

        src_node_idx, dst_node_idx = edges_tuple

        # To use as tensor indexes, these index tensors needs to be dtype=long.
        src_node_idx, dst_node_idx = src_node_idx.long(), dst_node_idx.long()

        return torch.norm(atom_positions[dst_node_idx] - atom_positions[src_node_idx], dim=1)
