"""
This is an implementation of GeoGNN using Pytorch/Pytorch Geometric.
"""

from torch import nn, Tensor
from SqrtGraphNorm import SqrtGraphNorm
from SimpleGIN import SimpleGIN
from FeaturesEmbedding import FeaturesEmbedding
from FeaturesRBF import FeaturesRBF
from Utils import Feature, FeatureName, RBFCenters, RBFGamma, Utils
from dgl import DGLGraph
from dgl.nn.pytorch.glob import AvgPooling
from typing import cast


class GeoGNNBlock(nn.Module):
    """
    GeoGNN Block
    """
    def __init__(self, embed_dim:int=32, dropout_rate:float=0.5, has_last_act:bool=True):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.has_last_act = has_last_act
        self.gnn = SimpleGIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = SqrtGraphNorm()
        if has_last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self) -> None:
        self.gnn.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, graph: DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tensor:
        out = self.gnn.forward(graph, node_feats, edge_feats)
        out = self.norm.forward(out)
        out = self.graph_norm.forward(graph, out)
        if self.has_last_act:
            out = self.act.forward(out)
        out = self.dropout.forward(out)
        out = out + node_feats
        return out


class GeoGNNLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float,
        has_last_act: bool,
        atom_feat_dict: dict[FeatureName, Feature],
        bond_feat_dict: dict[FeatureName, Feature],
        bond_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]],
        bond_angle_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] = Utils.RBF_PARAMS['bond_angle']
    ) -> None:
        super(GeoGNNLayer, self).__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.bond_embedding = FeaturesEmbedding(bond_feat_dict, embed_dim)
        self.bond_rbf = FeaturesRBF(bond_rbf_param_dict, embed_dim)
        self.bond_angle_rbf = FeaturesRBF(bond_angle_rbf_param_dict, embed_dim)
        self.atom_bond_gnn_block = GeoGNNBlock(embed_dim, dropout_rate, has_last_act)
        self.bond_angle_gnn_block = GeoGNNBlock(embed_dim, dropout_rate, has_last_act)

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
        edge_feats: Tensor
    ) -> tuple[Tensor, Tensor]:
        node_out = self.atom_bond_gnn_block.forward(atom_bond_graph, node_feats, edge_feats)

        bond_embed = self.bond_embedding.forward(atom_bond_graph.edata) \
            + self.bond_rbf.forward(atom_bond_graph.edata)

        # Since `atom_bond_graph` is bidirected, there's 2 copies of each edge
        # (ie. the bonds). This removes one of the bond edge copies, so as to
        # match the number of bond nodes in `bond_angle_graph`.
        bond_embed = GeoGNNLayer._get_unidirected_feats(atom_bond_graph, bond_embed)

        bond_angle_embed = self.bond_angle_rbf.forward(bond_angle_graph.edata)
        edge_out = self.bond_angle_gnn_block.forward(bond_angle_graph, bond_embed, bond_angle_embed)

        return node_out, edge_out

    @staticmethod
    def _get_unidirected_feats(
        bidirected_graph: DGLGraph,
        bidirected_edge_feats: Tensor
    ) -> Tensor:
        """
        Converts bi-directed edge features to uni-directed. Bi-directed graphs
        have 2 copies of each undirected-edge; this method removes the values of
        1 of those copies.
        """
        u, v = bidirected_graph.edges()

        # Include only edge features where
        # the edge's source-node ID < the destination-node ID.
        mask = u < v
        return bidirected_edge_feats[mask]


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
        atom_feat_dict: dict[FeatureName, Feature] = Utils.FEATURES['atom_feats'],
        bond_feat_dict: dict[FeatureName, Feature] = Utils.FEATURES['bond_feats'],
        bond_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] = Utils.RBF_PARAMS['bond'],
        bond_angle_rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]] = Utils.RBF_PARAMS['bond_angle']
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
                Defaults to 0.2.
            num_of_layers (int, optional): Number of `GeoGNNLayer` layers used. \
                Defaults to 8.
            atom_feat_dict (dict[FeatureName, Feature], optional): Details for \
                the atom features. Defaults to Utils.FEATURES['atom_feats'].
            bond_feat_dict (dict[FeatureName, Feature], optional): Details for \
                the bond features. Defaults to Utils.FEATURES['bond_feats'].
            bond_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]], optional): \
                RBF-layer's params for the bonds. Defaults to Utils.RBF_PARAMS['bond'].
            bond_angle_rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]], optional): \
                RBF-layer's params for the bond-angles. Defaults to Utils.RBF_PARAMS['bond_angle'].
        """
        super(GeoGNNModel, self).__init__()

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
        bond_angle_graph: DGLGraph
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_embeddings = self.init_atom_embedding.forward(atom_bond_graph.ndata)
        edge_embeddings = self.init_bond_embedding.forward(atom_bond_graph.edata) \
            + self.init_bond_rbf.forward(atom_bond_graph.edata)

        node_out = node_embeddings
        edge_out = edge_embeddings
        for gnn_layer in self.gnn_layer_list:
            cast(GeoGNNLayer, gnn_layer).forward(atom_bond_graph, bond_angle_graph, node_out, edge_out)

        graph_repr = self.graph_pool.forward(atom_bond_graph, node_out)
        return node_out, edge_out, graph_repr


# class GeoPredModel(nn.Layer):
#     """tbd"""
#     def __init__(self, model_config, compound_encoder):
#         super(GeoPredModel, self).__init__()
#         self.compound_encoder = compound_encoder
        
#         self.hidden_size = model_config['hidden_size']
#         self.dropout_rate = model_config['dropout_rate']
#         self.act = model_config['act']
#         self.pretrain_tasks = model_config['pretrain_tasks']
        
#         # context mask
#         if 'Cm' in self.pretrain_tasks:
#             self.Cm_vocab = model_config['Cm_vocab']
#             self.Cm_linear = nn.Linear(compound_encoder.embed_dim, self.Cm_vocab + 3)
#             self.Cm_loss = nn.CrossEntropyLoss()
#         # functinal group
#         self.Fg_linear = nn.Linear(compound_encoder.embed_dim, model_config['Fg_size']) # 494
#         self.Fg_loss = nn.BCEWithLogitsLoss()
#         # bond angle with regression
#         if 'Bar' in self.pretrain_tasks:
#             self.Bar_mlp = MLP(2,
#                     hidden_size=self.hidden_size,
#                     act=self.act,
#                     in_size=compound_encoder.embed_dim * 3,
#                     out_size=1,
#                     dropout_rate=self.dropout_rate)
#             self.Bar_loss = nn.SmoothL1Loss()
#         # bond length with regression
#         if 'Blr' in self.pretrain_tasks:
#             self.Blr_mlp = MLP(2,
#                     hidden_size=self.hidden_size,
#                     act=self.act,
#                     in_size=compound_encoder.embed_dim * 2,
#                     out_size=1,
#                     dropout_rate=self.dropout_rate)
#             self.Blr_loss = nn.SmoothL1Loss()
#         # atom distance with classification
#         if 'Adc' in self.pretrain_tasks:
#             self.Adc_vocab = model_config['Adc_vocab']
#             self.Adc_mlp = MLP(2,
#                     hidden_size=self.hidden_size,
#                     in_size=self.compound_encoder.embed_dim * 2,
#                     act=self.act,
#                     out_size=self.Adc_vocab + 3,
#                     dropout_rate=self.dropout_rate)
#             self.Adc_loss = nn.CrossEntropyLoss()

#         print('[GeoPredModel] pretrain_tasks:%s' % str(self.pretrain_tasks))

#     def _get_Cm_loss(self, feed_dict, node_repr):
#         masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
#         logits = self.Cm_linear(masked_node_repr)
#         loss = self.Cm_loss(logits, feed_dict['Cm_context_id'])
#         return loss

#     def _get_Fg_loss(self, feed_dict, graph_repr):
#         fg_label = paddle.concat(
#                 [feed_dict['Fg_morgan'], 
#                 feed_dict['Fg_daylight'], 
#                 feed_dict['Fg_maccs']], 1)
#         logits = self.Fg_linear(graph_repr)
#         loss = self.Fg_loss(logits, fg_label)
#         return loss

#     def _get_Bar_loss(self, feed_dict, node_repr):
#         node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
#         node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
#         node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
#         node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
#         pred = self.Bar_mlp(node_ijk_repr)
#         loss = self.Bar_loss(pred, feed_dict['Ba_bond_angle'] / np.pi)
#         return loss

#     def _get_Blr_loss(self, feed_dict, node_repr):
#         node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
#         node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
#         node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
#         pred = self.Blr_mlp(node_ij_repr)
#         loss = self.Blr_loss(pred, feed_dict['Bl_bond_length'])
#         return loss

#     def _get_Adc_loss(self, feed_dict, node_repr):
#         node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
#         node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
#         node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
#         logits = self.Adc_mlp.forward(node_ij_repr)
#         atom_dist = paddle.clip(feed_dict['Ad_atom_dist'], 0.0, 20.0)
#         atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Adc_vocab, 'int64')
#         loss = self.Adc_loss(logits, atom_dist_id)
#         return loss

#     def forward(self, graph_dict, feed_dict, return_subloss=False):
#         """
#         Build the network.
#         """
#         node_repr, edge_repr, graph_repr = self.compound_encoder.forward(
#                 graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'])
#         masked_node_repr, masked_edge_repr, masked_graph_repr = self.compound_encoder.forward(
#                 graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'])

#         sub_losses = {}
#         if 'Cm' in self.pretrain_tasks:
#             sub_losses['Cm_loss'] = self._get_Cm_loss(feed_dict, node_repr)
#             sub_losses['Cm_loss'] += self._get_Cm_loss(feed_dict, masked_node_repr)
#         if 'Fg' in self.pretrain_tasks:
#             sub_losses['Fg_loss'] = self._get_Fg_loss(feed_dict, graph_repr)
#             sub_losses['Fg_loss'] += self._get_Fg_loss(feed_dict, masked_graph_repr)
#         if 'Bar' in self.pretrain_tasks:
#             sub_losses['Bar_loss'] = self._get_Bar_loss(feed_dict, node_repr)
#             sub_losses['Bar_loss'] += self._get_Bar_loss(feed_dict, masked_node_repr)
#         if 'Blr' in self.pretrain_tasks:
#             sub_losses['Blr_loss'] = self._get_Blr_loss(feed_dict, node_repr)
#             sub_losses['Blr_loss'] += self._get_Blr_loss(feed_dict, masked_node_repr)
#         if 'Adc' in self.pretrain_tasks:
#             sub_losses['Adc_loss'] = self._get_Adc_loss(feed_dict, node_repr)
#             sub_losses['Adc_loss'] += self._get_Adc_loss(feed_dict, masked_node_repr)

#         loss = 0
#         for name in sub_losses:
#             loss += sub_losses[name]
#         if return_subloss:
#             return loss, sub_losses
#         else:
#             return loss