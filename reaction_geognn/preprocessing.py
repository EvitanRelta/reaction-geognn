"""
Data-preprocesing related stuff.
"""

from typing import Mapping, cast

import dgl, torch
from dgl import DGLGraph
from geognn import Preprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor


def reaction_smart_to_graph(
    reaction_smart: str,
    device: torch.device = torch.device('cpu')
) -> tuple[DGLGraph, DGLGraph]:
    """
    Converts a balanced and atom-mapped reaction's SMART string into 2 `DGLGraphs`:
    - a graph with atoms as nodes, bonds as edges
    - a graph with bonds as nodes, bond-angles as edges

    The product's nodes/features are concatenated behind the reactant's
    node/features in the graphs.

    For the atom-bond graph, the nodes (ie. the atoms) are arranged according to
    their atom-mapping number.
    (ie. reactant atom with atom-mapping 1 is node 0 in the graph, since
    atom-mappings are one-indexed while nodes are zero-indexed)

    Args:
        reaction_smart (str): A balanced and atom-mapped reaction's SMILES string.
        device (torch.device, optional): The CPU/GPU to set returned graphs to use. \
            Defaults to torch.device('cpu').

    Returns:
        tuple[DGLGraph, DGLGraph]: 1st graph is the atom-bond graph, 2nd \
            is the bond-angle graph.
    """
    # `reactant_smiles` contains all reactants, `product_smiles` contains all products.
    reactant_smiles, product_smiles = reaction_smart.split('>>')
    _validate_smiles(reactant_smiles, product_smiles)

    # Create individual graphs for each molecule and then batch them together
    reactant_atom_bond_graph, reactant_bond_angle_graph \
        = Preprocessing.smiles_to_graphs(reactant_smiles, device=device)
    product_atom_bond_graph, product_bond_angle_graph \
        = Preprocessing.smiles_to_graphs(product_smiles, device=device)

    # `.ndata["_is_reactant"]` is used for splitting reactants features from
    # products in the downstream model.
    num_of_atoms = reactant_atom_bond_graph.num_nodes()
    reactant_atom_bond_graph.ndata['_is_reactant'] \
        = torch.ones(num_of_atoms, dtype=torch.bool, device=device)
    product_atom_bond_graph.ndata['_is_reactant'] \
        = torch.zeros(num_of_atoms, dtype=torch.bool, device=device)

    return (
        _merge_graphs([reactant_atom_bond_graph, product_atom_bond_graph]),
        _merge_graphs([reactant_bond_angle_graph, product_bond_angle_graph]),
    )


def _validate_smiles(reactant_smiles: str, product_smiles: str) -> None:
    """
    Validate that the SMILES are:
    - balanced (num of reactant atoms == num of product atoms).
    - have valid atom-mappings.

    Args:
        reactant_smiles (str): SMILES string of all the reactants.
        product_smiles (str): SMILES string of all the products.
    """
    # Prevents `AllChem.MolFromSmiles` from removing the atom-mapped hydrogens in the SMILES.
    # Based on: https://github.com/rdkit/rdkit/discussions/4703#discussioncomment-1656372
    smiles_parser = Chem.SmilesParserParams() # type: ignore
    smiles_parser.removeHs = False

    reactant_mol = AllChem.MolFromSmiles(reactant_smiles, smiles_parser) # type: ignore
    product_mol = AllChem.MolFromSmiles(product_smiles, smiles_parser) # type: ignore

    assert reactant_mol.GetNumAtoms() == product_mol.GetNumAtoms(), \
        'Reaction SMART is not balanced.'

    reactant_atom_map = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
    product_atom_map = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
    assert len(reactant_atom_map) == len(set(reactant_atom_map)), \
        'Reactants have duplicate atom mapping numbers.'
    assert len(product_atom_map) == len(set(product_atom_map)), \
        'Products have duplicate atom mapping numbers.'
    assert max(reactant_atom_map) == max(product_atom_map) == reactant_mol.GetNumAtoms(), \
        "Atom mappings numbers don't match number of atoms."


def _merge_graphs(graph_list: list[DGLGraph]) -> DGLGraph:
    """
    Merge multiple graphs into a single graph. Similar to `dgl.batch` but it
    it doesn't "batch" the graphs.
    """
    device = graph_list[0].device

    # Initialize empty lists for the new nodes and edges
    new_edges_src: list[int] = []
    new_edges_dst: list[int] = []
    new_node_features: dict[str, list[Tensor]] = {}
    new_edge_features: dict[str, list[Tensor]] = {}

    # Track how many nodes we've added so far
    num_nodes_so_far: int = 0

    for g in graph_list:
        # Adjust the edge list by the number of nodes we've added
        edges = g.edges()
        edges = cast(tuple[Tensor, Tensor], edges)
        edges = (edges[0] + num_nodes_so_far, edges[1] + num_nodes_so_far)

        # Add the edges to our new edge list
        new_edges_src.extend(edges[0].tolist())
        new_edges_dst.extend(edges[1].tolist())

        # Add the node features to our new node feature dict
        for feature_name in cast(Mapping[str, Tensor], g.ndata):
            if feature_name not in new_node_features:
                new_node_features[feature_name] = []
            new_node_features[feature_name].extend(g.ndata[feature_name])

        # Add the edge features to our new edge feature dict
        for feature_name in cast(Mapping[str, Tensor],g.edata):
            if feature_name not in new_edge_features:
                new_edge_features[feature_name] = []
            new_edge_features[feature_name].extend(g.edata[feature_name])

        # Update the total number of nodes we've seen
        num_nodes_so_far += cast(int, g.number_of_nodes())

    # Create the new graph
    new_g = dgl.graph((new_edges_src, new_edges_dst), num_nodes=num_nodes_so_far, device=device)

    # Add the node features to the new graph
    for feature_name, features in new_node_features.items():
        new_g.ndata[feature_name] = torch.tensor(features, device=device)

    # Add the edge features to the new graph
    for feature_name, features in new_edge_features.items():
        new_g.edata[feature_name] = torch.tensor(features, device=device)

    return new_g
