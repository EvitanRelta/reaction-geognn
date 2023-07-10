"""Utility functions for manipulating DGLGraphs."""

from typing import Mapping, cast

import dgl, torch
from dgl import DGLGraph
from torch import Tensor


def split_batched_data(batched_node_repr: Tensor, batched_atom_bond_graph: DGLGraph) -> list[tuple[Tensor, DGLGraph]]:
    """Split batched node feature/representation tensor and `DGLGraph` into
    individual node-feat/repr tensors and graphs.

    Args:
        batched_node_repr (Tensor): Batched node-feature/representation, \
            size `(total_num_nodes, feat_size)`.
        batched_atom_bond_graph (DGLGraph): Batched atom-bond graph, where the \
            nodes are atoms, edges are bonds.

    Returns:
        list[tuple[Tensor, DGLGraph]]: List of individual node-feat/repr and \
            graphs in the batch.
    """
    output: list[tuple[Tensor, DGLGraph]] = []
    start_index = 0
    for graph in dgl.unbatch(batched_atom_bond_graph):
        num_nodes = graph.number_of_nodes()
        node_repr = batched_node_repr[start_index : start_index + num_nodes]
        start_index += num_nodes
        output.append((node_repr, graph))
    return output


def split_reactant_product_nodes(node_repr: Tensor, atom_bond_graph: DGLGraph) -> tuple[Tensor, Tensor]:
    """Split reactant's node-feature/representation from product's.

    Args:
        node_repr (Tensor): Combined reactant and product node-feat/repr.
        atom_bond_graph (DGLGraph): Atom-bond graph containing both the reactant \
            and product subgraphs.

    Returns:
        tuple[Tensor, Tensor]: Reactant's and product's node-feat/repr: \
            `(reactant_node_repr, product_node_repr)`
    """
    assert '_is_reactant' in atom_bond_graph.ndata, \
        'Atom-bond graphs needs to have .ndata["_is_reactant"] of dtype=bool.'
    assert len(node_repr) % 2 == 0, 'Odd number of nodes in node_repr.'

    mask = atom_bond_graph.ndata['_is_reactant']
    assert isinstance(mask, Tensor) and mask.dtype == torch.bool
    reactant_node_repr = node_repr[mask]
    product_node_repr = node_repr[~mask]

    assert len(reactant_node_repr) == len(node_repr) // 2
    assert len(product_node_repr) == len(node_repr) // 2
    return reactant_node_repr, product_node_repr


def concat_graphs(graph_list: list[DGLGraph]) -> DGLGraph:
    """
    Merge multiple graphs into a single graph by concatenating their nodes,
    edges and features. Similar to `dgl.batch` but it doesn't "batch" the graphs.
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
