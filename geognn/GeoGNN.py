"""
This is an implementation of GeoGNN using PyTorch/PyTorch Geometric.
"""

from typing import Literal, overload

import torch
from dgl import DGLGraph
from dgl.nn.pytorch.glob import AvgPooling
from torch import Tensor, nn

from .layers import FeaturesEmbedding, FeaturesRBF, SimpleGIN, SqrtGraphNorm
from .Preprocessing import Feature, FeatureName, Preprocessing, RBFCenters, \
    RBFGamma


class InnerGNN(nn.Module):
    """
    The GNN used inside of `GeoGNNModel`.

    This is the "GNN" part of GeoGNN, including normalisation layers but
    excluding feature embedding layers.
    """

    def __init__(
        self,
        in_feat_size: int,
        hidden_size: int,
        out_feat_size: int,
        dropout_rate: float,
        has_last_act: bool = True,
    ):
        """
        Args:
            in_feat_size (int): The size of each feature in the graph \
                (if the feats were encoded into embeddings, this'll be the embedding size).
            hidden_size (int): Hidden layer's size of the MLP \
                (the MLP after message-passing).
            out_feat_size (int): The output size for each feature.
            dropout_rate (float): Dropout rate for the dropout layers.
            has_last_act (bool, optional): Whether to pass the final output \
                through an activation function (ie. ReLU). Defaults to True.
        """
        super().__init__()

        self.has_last_act = has_last_act
        self.gnn = SimpleGIN(in_feat_size, hidden_size, out_feat_size)
        self.norm = nn.LayerNorm(out_feat_size)
        self.graph_norm = SqrtGraphNorm()
        if has_last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self) -> None:
        self.gnn.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, graph: DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tensor:
        """
        Args:
            graph (DGLGraph): The input graph, where each node/edge feature is \
                of size `in_feat_size`, as defined in the constructor.
            node_feats (Tensor): The input node features, size `(num_of_nodes, in_feat_size)`, \
                where `in_feat_size` is defined in the constructor.
            edge_feats (Tensor): The input edge features, size `(num_of_edges, in_feat_size)`, \
                where `in_feat_size` is defined in the constructor.

        Returns:
            Tensor: Output features that incorporates both node and edge features, \
                size `(num_of_nodes, out_feat_size)`, where `out_feat_size` is \
                defined in the constructor.
        """
        out = self.gnn.forward(graph, node_feats, edge_feats)
        out = self.norm.forward(out)
        out = self.graph_norm.forward(graph, out)
        if self.has_last_act:
            out = self.act.forward(out)
        out = self.dropout.forward(out)
        out = out + node_feats
        return out


class GeoGNNLayer(nn.Module):
    """
    A single GeoGNN layer.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float,
        has_last_act: bool,
        atom_feat_dict: dict[FeatureName, Feature],
        bond_feat_dict: dict[FeatureName, Feature],
        bond_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]],
        bond_angle_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] = Preprocessing.RBF_PARAMS['bond_angle'],
    ) -> None:
        """
        Args:
            embed_dim (int): Dimension of the feature embeddings.
            dropout_rate (float): Dropout rate for the dropout layers.
            has_last_act (bool): Whether to pass the final output through an \
                activation function (ie. ReLU).
            atom_feat_dict (dict[FeatureName, Feature]): Details for the atom features.
            bond_feat_dict (dict[FeatureName, Feature]): Details for the bond features.
            bond_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]]): \
                RBF-layer's params for the bonds.
            bond_angle_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]]): \
                RBF-layer's params for the bond-angles.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.bond_embedding = FeaturesEmbedding(bond_feat_dict, embed_dim)
        self.bond_rbf = FeaturesRBF(bond_rbf_param_dict, embed_dim)
        self.bond_angle_rbf = FeaturesRBF(bond_angle_rbf_param_dict, embed_dim)
        self.atom_bond_gnn_block = InnerGNN(
            in_feat_size = embed_dim,
            hidden_size = embed_dim * 2,
            out_feat_size = embed_dim,
            dropout_rate = dropout_rate,
            has_last_act = has_last_act,
        )
        self.bond_angle_gnn_block = InnerGNN(
            in_feat_size = embed_dim,
            hidden_size = embed_dim * 2,
            out_feat_size = embed_dim,
            dropout_rate = dropout_rate,
            has_last_act = has_last_act,
        )

    def reset_parameters(self) -> None:
        self.bond_embedding.reset_parameters()
        self.bond_rbf.reset_parameters()
        self.bond_angle_rbf.reset_parameters()
        self.atom_bond_gnn_block.reset_parameters()
        self.bond_angle_gnn_block.reset_parameters()

    def forward(
        self,
        atom_bond_graph: DGLGraph,
        bond_angle_graph: DGLGraph,
        node_feats: Tensor,
        edge_feats: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.
            node_feats (Tensor): The input node features, \
                size `(num_of_nodes, self.embed_dim)`.
            edge_feats (Tensor): The input edge features, \
                size `(num_of_edges, self.embed_dim)`.

        Returns:
            tuple[Tensor, Tensor]: The node and edge representations in \
                the form - `(node_repr, edge_repr)`
        """
        node_out = self.atom_bond_gnn_block.forward(atom_bond_graph, node_feats, edge_feats)

        bond_embed = self.bond_embedding.forward(atom_bond_graph.edata) \
            + self.bond_rbf.forward(atom_bond_graph.edata)

        # Since `atom_bond_graph` is bidirected, there's 2 copies of each edge (ie. the bonds),
        # where the forward and backward edges were interleaved during the preprocessing.
        # (ie. [edge_1, edge_opp_1, edge_2, edge_opp_2, ...])
        # This removes one of the bond edge copies, so as to match the number of
        # bond nodes in `bond_angle_graph`.
        bond_embed = bond_embed[::2]

        bond_angle_embed = self.bond_angle_rbf.forward(bond_angle_graph.edata)
        edge_out = self.bond_angle_gnn_block.forward(bond_angle_graph, bond_embed, bond_angle_embed)

        # This re-adds the bond-edge copies that was removed above.
        edge_out = edge_out.repeat_interleave(2, dim=0)
        return node_out, edge_out


class GeoGNNModel(nn.Module):
    """
    The GeoGNN Model used in GEM.
    """

    def __init__(
        self,
        embed_dim: int = 32,

        # Pretraining's dropout rate is 0.2, based on `pretrain.sh` script:
        # https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/pretrain.sh#L26
        dropout_rate: float = 0.5,

        num_of_layers: int = 8,
        atom_feat_dict: dict[FeatureName, Feature] = Preprocessing.FEATURES['atom_feats'],
        bond_feat_dict: dict[FeatureName, Feature] = Preprocessing.FEATURES['bond_feats'],
        bond_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] \
            = Preprocessing.RBF_PARAMS['bond'],
        bond_angle_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] \
            = Preprocessing.RBF_PARAMS['bond_angle'],
    ) -> None:
        """
        Default values for `embed_dim`, `dropout_rate` and `num_of_layers` and
        the `self.graph_pool` value are based on GeoGNN's `geognn_l8.json`
        config:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json

        ### Note:
        `dropout_rate` during GeoGNN's pretraining is `0.2` based on
        `pretrain.sh`:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/pretrain.sh

        Args:
            embed_dim (int, optional): Dimension of the feature embeddings. \
                Defaults to 32.
            dropout_rate (float, optional): Dropout rate for the dropout layers. \
                Defaults to 0.5.
            num_of_layers (int, optional): Number of `GeoGNNLayer` layers used. \
                Defaults to 8.
            atom_feat_dict (dict[FeatureName, Feature], optional): Details for \
                the atom features. Defaults to Preprocessing.FEATURES['atom_feats'].
            bond_feat_dict (dict[FeatureName, Feature], optional): Details for \
                the bond features. Defaults to Preprocessing.FEATURES['bond_feats'].
            bond_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]], optional): \
                RBF-layer's params for the bonds. Defaults to Preprocessing.RBF_PARAMS['bond'].
            bond_angle_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]], optional): \
                RBF-layer's params for the bond-angles. Defaults to Preprocessing.RBF_PARAMS['bond_angle'].
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_of_layers = num_of_layers

        self.init_atom_embedding = FeaturesEmbedding(atom_feat_dict, embed_dim)
        self.init_bond_embedding = FeaturesEmbedding(bond_feat_dict, embed_dim)
        self.init_bond_rbf = FeaturesRBF(bond_rbf_param_dict, embed_dim)

        is_not_last_layer = lambda layer_idx: layer_idx != num_of_layers
        dicts = (atom_feat_dict, bond_feat_dict, bond_rbf_param_dict, bond_angle_rbf_param_dict)
        self.gnn_layer_list = nn.ModuleList([
            GeoGNNLayer(embed_dim, dropout_rate, is_not_last_layer(i), *dicts) \
                for i in range(num_of_layers)
        ])

        self.graph_pool = AvgPooling()

    def reset_parameters(self) -> None:
        self.init_atom_embedding.reset_parameters()
        self.init_bond_embedding.reset_parameters()
        self.init_bond_rbf.reset_parameters()

        for gnn_layer in self.gnn_layer_list:
            assert isinstance(gnn_layer, GeoGNNLayer)
            gnn_layer.reset_parameters()

    @overload
    def forward(self, atom_bond_graph: DGLGraph, bond_angle_graph: DGLGraph, pool_graph: Literal[True] = True) -> tuple[Tensor, Tensor, Tensor]: ...
    @overload
    def forward(self, atom_bond_graph: DGLGraph, bond_angle_graph: DGLGraph, pool_graph: Literal[False]) -> tuple[Tensor, Tensor]: ...
    def forward(
        self,
        atom_bond_graph: DGLGraph,
        bond_angle_graph: DGLGraph,
        pool_graph: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        """
        Args:
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.
            pool_graph (bool): Whether to pool and return the graph representation. \
                Defaults to True.

        Returns:
            tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]: The node, \
                edge (and optionally graph) representations in the form \
                - `(node_repr, edge_repr, graph_repr)`
        """
        node_embeddings = self.init_atom_embedding.forward(atom_bond_graph.ndata)
        edge_embeddings = self.init_bond_embedding.forward(atom_bond_graph.edata) \
            + self.init_bond_rbf.forward(atom_bond_graph.edata)

        node_out = node_embeddings
        edge_out = edge_embeddings
        for gnn_layer in self.gnn_layer_list:
            assert isinstance(gnn_layer, GeoGNNLayer)
            node_out, edge_out = gnn_layer.forward(atom_bond_graph, bond_angle_graph, node_out, edge_out)

        if not pool_graph:
            return node_out, edge_out

        graph_repr = self.graph_pool.forward(atom_bond_graph, node_out)
        return node_out, edge_out, graph_repr
