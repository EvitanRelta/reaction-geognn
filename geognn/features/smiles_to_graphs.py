from typing import Literal, TypeAlias, overload

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.transforms.functional import add_reverse_edges, to_simple
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms as rdmt  # type: ignore
from torch import Tensor

from .features import FLOAT_ATOM_FEATURES, FLOAT_BOND_FEATURES, \
    LABEL_ENCODED_ATOM_FEATURES, LABEL_ENCODED_BOND_FEATURES, AtomPosition
from .rdkit_types import Conformer, Mol

FeatureCategory: TypeAlias = Literal['atom_feats', 'bond_feats']
FeatureName: TypeAlias = str

RBFFeatureCategory: TypeAlias = Literal['bond', 'bond_angle']
RBFCenters: TypeAlias = Tensor
"""1D tensor of all the RBF centers for a feature."""
RBFGamma: TypeAlias = float
"""Hyperparameter for controlling the spread of the RBF's Gaussian basis function."""


@overload
def smiles_to_graphs(smiles: str, device: torch.device, return_mol_conf: Literal[True]) ->  tuple[DGLGraph, DGLGraph, Mol, Conformer]: ...
@overload
def smiles_to_graphs(smiles: str, device: torch.device, return_mol_conf: Literal[False] = False) -> tuple[DGLGraph, DGLGraph]: ...
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
    mol = AllChem.MolFromSmiles(smiles)                                         # type: ignore
    mol, conf = _generate_conformer(mol)

    atom_bond_graph = _get_atom_bond_graph(mol, conf, device)
    bond_angle_graph = _get_bond_angle_graph(mol, conf, device)
    if return_mol_conf:
        return atom_bond_graph, bond_angle_graph, mol, conf
    return atom_bond_graph, bond_angle_graph


def _to_bidirected_copy(g: DGLGraph) -> DGLGraph:
    """
    Exactly the same as `dgl.to_bidirected`, but copies both node and edge
    features.

    Args:
        g (DGLGraph): The input directed graph.

    Returns:
        DGLGraph: Graph `g` but bidirected and with copied node/edge features.
    """
    g = add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g = to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)      # type: ignore
    return g


def _generate_conformer(mol: Mol, numConfs: int = 10) -> tuple[Mol, Conformer]:
    new_mol = Chem.AddHs(mol)                                                   # type: ignore
    res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)                # type: ignore
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)                            # type: ignore
    new_mol = Chem.RemoveHs(new_mol)                                            # type: ignore
    index = np.argmin([x[1] for x in res])
    conf = new_mol.GetConformer(id=int(index))
    return new_mol, conf


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
    for i, bond in enumerate(mol.GetBonds()):
        edges[0][i] = bond.GetBeginAtomIdx()
        edges[1][i] = bond.GetEndAtomIdx()
    graph = dgl.graph(edges, num_nodes=mol.GetNumAtoms(), idtype=torch.int32)

    # Add node features.
    for feat in LABEL_ENCODED_ATOM_FEATURES + FLOAT_ATOM_FEATURES:
        graph.ndata[feat.name] = feat.get_feat_values(mol, conf, graph)

    # Temp feat used for generating bond lengths (edge feat).
    temp_feat = AtomPosition('_atom_pos')
    graph.ndata[temp_feat.name] = temp_feat.get_feat_values(mol, conf, graph)

    # Add edge features.
    for feat in LABEL_ENCODED_BOND_FEATURES + FLOAT_BOND_FEATURES:
        graph.edata[feat.name] = feat.get_feat_values(mol, conf, graph)

    # Remove temporary feat used in computing other feats.
    del graph.ndata['_atom_pos']

    graph = _to_bidirected_copy(graph)   # Convert to undirected graph.
    graph = graph.to(device)    # Move graph to CPU/GPU depending on `device`.
    return graph


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
