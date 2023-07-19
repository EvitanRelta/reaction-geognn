"""Utility functions for manipulating DGLGraphs."""

from typing import Mapping, cast

import dgl, torch
from dgl import DGLGraph
from torch import Tensor


def split_batched_data(
    *,
    batched_atom_bond_graph: DGLGraph,
    batched_bond_angle_graph: DGLGraph | None = None,
    batched_node_repr: Tensor | None = None,
    batched_edge_repr: Tensor | None = None,
) -> list[tuple[DGLGraph | Tensor, ...]]:

    """Split batched graph(s) and/or node/edge-representation tensors into
    individual graphs and tensors.

    Args:
        batched_atom_bond_graph (DGLGraph): Batched atom-bond graph, \
            where the nodes are atoms, edges are bonds.
        batched_bond_angle_graph (DGLGraph, optional): Batched bond-angle graph, \
            where the nodes are bonds, edges are bond-angles.
        batched_node_repr (Tensor, optional): Batched node-representation, \
            size `(total_num_nodes, feat_size)`.
        batched_edge_repr (Tensor, optional): Batched edge-representation, \
            size `(total_num_edges, feat_size)`.

    Returns:
        list[tuple[DGLGraph | Tensor, ...]]: Unbatched graphs/representation-tensors \
            in the order: `atom_bond_graph`, `bond_angle_graph`, `node_repr`, `edge_repr`. """
    output_lists: list[list[DGLGraph] | tuple[Tensor]] = []
    output_lists.append(dgl.unbatch(batched_atom_bond_graph))

    if batched_bond_angle_graph != None:
        output_lists.append(dgl.unbatch(batched_bond_angle_graph))

    if batched_node_repr != None:
        batch_num_nodes_list: list[int] = batched_atom_bond_graph.batch_num_nodes().tolist()
        node_repr_list = batched_node_repr.split(batch_num_nodes_list)
        assert isinstance(node_repr_list, tuple)
        output_lists.append(node_repr_list)

    if batched_edge_repr != None:
        batch_num_edges_list: list[int] = batched_atom_bond_graph.batch_num_edges().tolist()
        edge_repr_list = batched_edge_repr.split(batch_num_edges_list)
        assert isinstance(edge_repr_list, tuple)
        output_lists.append(edge_repr_list)

    return list(zip(*output_lists)) # type: ignore


def split_reactant_product_node_feat(node_repr: Tensor, atom_bond_graph: DGLGraph) -> tuple[Tensor, Tensor]:
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
