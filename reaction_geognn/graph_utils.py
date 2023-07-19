"""Utility functions for manipulating DGLGraphs."""

from typing import Mapping, cast

import dgl, torch
from dgl import DGLGraph
from torch import Tensor


def split_batched_data(
    *,
    batched_atom_bond_graph: DGLGraph,
    batched_bond_angle_graph: DGLGraph | None = None,
    batched_superimposed_atom_graph: DGLGraph | None = None,
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
            in the order: `atom_bond_graph`, `bond_angle_graph`, \
            `superimposed_atom_graph`, `node_repr`, `edge_repr`.
    """
    output_lists: list[list[DGLGraph] | tuple[Tensor]] = []
    output_lists.append(dgl.unbatch(batched_atom_bond_graph))

    if batched_bond_angle_graph != None:
        output_lists.append(dgl.unbatch(batched_bond_angle_graph))
    if batched_superimposed_atom_graph != None:
        output_lists.append(dgl.unbatch(batched_superimposed_atom_graph))

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


# TODO: Add `bond_angle_graph.ndata["_is_reactant"]`, and remove
# `is_reactant_node_mask` in `split_reactant_product_graphs`.
# (and maybe copy the bond feats from `atom_bond_graph` -> `bond_angle_graph` so
# that `bond_angle_graph.ndata` doesn't just have ["_is_reactant"]`, which is weird)
def split_reactant_product_graphs(
    graph: DGLGraph,
    is_reactant_node_mask: Tensor | None = None,
) -> tuple[DGLGraph, DGLGraph]:
    """Split the reactant and product subgraphs.

    Args:
        graph (DGLGraph): Combined graph containing the reactant and product subgraphs.
        is_reactant_node_mask (Tensor | None, optional): The node masks for whether \
            a node in `graph` is from reactant. If specified, this will be used \
            instead of `graph.ndata['_is_reactant']`. Defaults to None.

    Returns:
        tuple[DGLGraph, DGLGraph]: Reactant and product subgraphs: \
            `(reactant_graph, product_graph)`
    """
    if is_reactant_node_mask is None:
        assert '_is_reactant' in graph.ndata, \
            'Graph needs to have `.ndata["_is_reactant"]` of `dtype=bool`.'
        is_reactant_node_mask = cast(Tensor, graph.ndata['_is_reactant'])

    assert isinstance(is_reactant_node_mask, Tensor)
    assert is_reactant_node_mask.dtype == torch.bool
    assert len(is_reactant_node_mask) == graph.num_nodes()

    return (
        dgl.node_subgraph(graph, is_reactant_node_mask),
        dgl.node_subgraph(graph, ~is_reactant_node_mask),
    )


def union_edges(edges1: tuple[Tensor, Tensor], edges2: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Get the union of edges between `edges2` and `edges1`.

    (ie. if `A` is the set of edges in `edges1` and `B` is that in `edges2`, this
    will return `A-union-B` edges)

    Args:
        edges1 (tuple[Tensor, Tensor]): First set of edges in the form - `(U, V)` \
            where `U` are the edges' src-node indexes, `V` are the dst-node indexes.
        edges2 (tuple[Tensor, Tensor]): Second set of edges in the form - `(U, V)` \
            where `U` are the edges' src-node indexes, `V` are the dst-node indexes.

    Returns:
        tuple[Tensor, Tensor]: Union of edges in the form - `(U, V)` where `U` \
            are the edges' src-node indexes, `V` are the dst-node indexes.
    """
    # Convert the 1D tensors into 2D tensors
    edges1_2d = torch.stack(edges1, dim=1)
    edges2_2d = torch.stack(edges2, dim=1)

    # Concatenate the edges along the second dimension
    edges_2d = torch.cat((edges1_2d, edges2_2d), dim=0)

    # Get the unique edges
    unique_edges_2d = torch.unique(edges_2d, dim=0)
    return unique_edges_2d[:, 0], unique_edges_2d[:, 1]


def superimpose_reactant_products_graphs(atom_bond_graph: DGLGraph) -> DGLGraph:
    """Get a graph where reactant's and product's bonds (ie. graph-edges) are
    superimposed in a single graph (ie. returned graph has edges from both
    reactants and products), where each bond-feat in `atom_bond_graph` will be
    converted to 2 edge feats in the superimposed-graph:

    - `edata["r_FEAT_NAME"]` - reactant's bond feat
    - `edata["p_FEAT_NAME"]` - product's bond feat

    where `FEAT_NAME` is the bond-feat's name. If the reactant/product doesn't
    have that bond, `edata["r_FEAT_NAME"]`/ `edata["p_FEAT_NAME"]`
    will be a zero tensor.

    Args:
        atom_bond_graph (DGLGraph): Atom-bond graph, where the nodes are atoms \
            and edges are bonds.

    Returns:
        DGLGraph: The feature-less, superimposed graph.
    """
    reactant_graph, product_graph = split_reactant_product_graphs(atom_bond_graph)
    edge_union = union_edges(reactant_graph.edges(), product_graph.edges())
    num_atoms = reactant_graph.num_nodes()
    feat_names: list[str] = [x for x in reactant_graph.edata.keys() if x[0] != '_']

    output = dgl.graph(((), ()), num_nodes=num_atoms).to(atom_bond_graph.device)

    # Fill new edge-features with empty tensors.
    for feat in feat_names:
        output.edata[f'r_{feat}'] = torch.tensor([]).to(atom_bond_graph.edata[feat]) # type: ignore
        output.edata[f'p_{feat}'] = torch.tensor([]).to(atom_bond_graph.edata[feat]) # type: ignore

    for u, v in zip(*edge_union):
        r_id = _get_edge_id(reactant_graph, u, v)
        p_id = _get_edge_id(product_graph, u, v)

        edge_data: dict[str, Tensor] = {}
        for feat in feat_names:
            edge_data[f'r_{feat}'] = reactant_graph.edata[feat][r_id].unsqueeze(0) \
                if r_id is not None else torch.tensor([0]).to(atom_bond_graph.edata[feat]) # type: ignore
            edge_data[f'p_{feat}'] = product_graph.edata[feat][p_id].unsqueeze(0) \
                if p_id is not None else torch.tensor([0]).to(atom_bond_graph.edata[feat]) # type: ignore
        output.add_edges(u.item(), v.item(), edge_data)

    return output

def _get_edge_id(g: DGLGraph, u_id: Tensor, v_id: Tensor) -> int | None:
    try:
        return g.edge_ids(u_id, v_id).item()
    except:
        return None
