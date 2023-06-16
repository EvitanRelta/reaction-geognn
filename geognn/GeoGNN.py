"""
This is an implementation of GeoGNN using PyTorch/PyTorch Geometric.
"""

from torch import nn, Tensor
import torch
from .layers.SqrtGraphNorm import SqrtGraphNorm
from .layers.SimpleGIN import SimpleGIN
from .layers.FeaturesEmbedding import FeaturesEmbedding
from .layers.FeaturesRBF import FeaturesRBF
from .Preprocessing import Feature, FeatureName, RBFCenters, RBFGamma, Preprocessing
from dgl import DGLGraph
from dgl.nn.pytorch.glob import AvgPooling
from typing import cast


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

        # Since `atom_bond_graph` is bidirected, there's 2 copies of each edge
        # (ie. the bonds). This removes one of the bond edge copies, so as to
        # match the number of bond nodes in `bond_angle_graph`.
        bond_embed = GeoGNNLayer._get_unidirected_feats(atom_bond_graph, bond_embed)

        bond_angle_embed = self.bond_angle_rbf.forward(bond_angle_graph.edata)
        edge_out = self.bond_angle_gnn_block.forward(bond_angle_graph, bond_embed, bond_angle_embed)

        # This reverses the `GeoGNNLayer._get_unidirected_feats` above.
        edge_out = GeoGNNLayer._get_bidirected_feats(atom_bond_graph, edge_out)
        return node_out, edge_out

    @staticmethod
    def _get_unidirected_feats(
        bidirected_graph: DGLGraph,
        bidirected_edge_feats: Tensor,
    ) -> Tensor:
        """
        Converts bi-directed edge features to uni-directed. Bi-directed graphs
        have 2 copies of each undirected-edge; this method removes the values of
        1 of those copies.

        `GeoGNNLayer._get_bidirected_feats` performs the reversed operation.
        """
        u, v = bidirected_graph.edges()

        # Include only edge features where
        # the edge's source-node ID < the destination-node ID.
        mask = u < v
        return bidirected_edge_feats[mask]

    @staticmethod
    def _get_bidirected_feats(
        bidirected_graph: DGLGraph,
        unidirected_edge_feats: Tensor,
    ) -> Tensor:
        """
        Converts uni-directed edge features to bi-directed. Bi-directed graphs
        have 2 copies of each undirected-edge; this method duplicates the values
        of each undirected-edge.

        `GeoGNNLayer._get_unidirected_feats` performs the reversed operation.
        """
        (u, v) = cast(tuple[Tensor, Tensor], bidirected_graph.edges())

        # Indices of the edges in the bi-directed graph that correspond to the undirected edges
        undirected_indices = (u < v).nonzero(as_tuple=True)[0]

        # Indices of the edges in the bi-directed graph that are the reverse of the undirected edges
        reversed_indices = (v < u).nonzero(as_tuple=True)[0]

        # Initialize bidirected edge features with zeros
        bidirected_edge_feats = torch.zeros((len(u), unidirected_edge_feats.size(1)), device=unidirected_edge_feats.device, dtype=unidirected_edge_feats.dtype)

        # Assign undirected features to the corresponding edges in the bi-directed graph
        bidirected_edge_feats[undirected_indices] = unidirected_edge_feats

        # Duplicate features for the reversed edges
        bidirected_edge_feats[reversed_indices] = unidirected_edge_feats

        return bidirected_edge_feats


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
            cast(GeoGNNLayer, gnn_layer).reset_parameters()

    def forward(
        self,
        atom_bond_graph: DGLGraph,
        bond_angle_graph: DGLGraph,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The node, edge and graph \
                representations in the form - `(node_repr, edge_repr, graph_repr)`
        """
        node_embeddings = self.init_atom_embedding.forward(atom_bond_graph.ndata)
        edge_embeddings = self.init_bond_embedding.forward(atom_bond_graph.edata) \
            + self.init_bond_rbf.forward(atom_bond_graph.edata)

        node_out = node_embeddings
        edge_out = edge_embeddings
        for gnn_layer in self.gnn_layer_list:
            node_out, edge_out = cast(GeoGNNLayer, gnn_layer) \
                .forward(atom_bond_graph, bond_angle_graph, node_out, edge_out)

        graph_repr = self.graph_pool.forward(atom_bond_graph, node_out)
        return node_out, edge_out, graph_repr
