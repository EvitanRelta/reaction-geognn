from typing import Any, Protocol, overload

import torch
from base_classes import GeoGNNLightningModule, LoggedHyperParams
from dgl import DGLGraph
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from geognn import GeoGNNModel
from geognn.encoder_model import InnerGNN
from geognn.features import FLOAT_BOND_FEATURES, LABEL_ENCODED_BOND_FEATURES
from geognn.layers import DropoutMLP, FeaturesEmbedding, FeaturesRBF
from torch import Tensor, nn
from typing_extensions import override

from .graph_utils import split_batched_data, split_reactant_product_node_feat


class AggregationGNN(nn.Module):
    """Layer that aggregates the encoder's output node-representation with the
    initial bond-features to give a new node-representation.
    """

    def __init__(
        self,
        unconcat_embed_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        # Embeddings.
        self.bond_embedding = FeaturesEmbedding(LABEL_ENCODED_BOND_FEATURES, unconcat_embed_dim)
        self.bond_rbf = FeaturesRBF(FLOAT_BOND_FEATURES, unconcat_embed_dim)

        # GNN.
        concat_embed_dim = 2 * unconcat_embed_dim
        self.superimposed_gnn = InnerGNN(
            in_feat_size = concat_embed_dim,
            hidden_size = concat_embed_dim * 2,
            out_feat_size = concat_embed_dim,
            dropout_rate = dropout_rate,
            has_last_act = True,
        )

    def forward(
        self,
        superimposed_atom_graph: DGLGraph,
        superimposed_atom_repr: Tensor,
    ) -> Tensor:
        """
        Args:
            superimposed_atom_graph (DGLGraph): Atom-bond graph where reactant's \
                and product's bond-edges are superimposed in the same graph.
            superimposed_atom_repr (Tensor): The input atom representation for \
                the superimposed graph, size `(num_atoms, self.embed_dim)`.

        Returns:
            Tensor: New node representation, size `(num_atoms, self.embed_dim)`
        """
        reactant_edge_feats: dict[str, Tensor] = {
            feat[2:]: tensor \
                for feat, tensor in superimposed_atom_graph.edata.items() \
                    if feat[:2] == 'r_'
        }
        product_edge_feats: dict[str, Tensor] = {
            feat[2:]: tensor \
                for feat, tensor in superimposed_atom_graph.edata.items() \
                    if feat[:2] == 'p_'
        }

        # Embeddings from initial bond features, computed the same way as in GeoGNN.
        reactant_bond_embed = self.bond_embedding.forward(reactant_edge_feats) \
            + self.bond_rbf.forward(reactant_edge_feats) # size (2 * num_bonds, unconcat_embed_dim)
        product_bond_embed = self.bond_embedding.forward(product_edge_feats) \
            + self.bond_rbf.forward(product_edge_feats) # size (2 * num_bonds, unconcat_embed_dim)
        assert reactant_bond_embed.shape == product_bond_embed.shape

        # Concat reactant's embeddings with difference, similar to how CGR did it.
        concat_bond_embed = torch.cat((
            reactant_bond_embed,
            product_bond_embed - reactant_bond_embed,
        ), dim=1) # size (num_bonds, 2 * unconcat_embed_dim)


        # Get new node-representation.
        assert superimposed_atom_repr.shape[0] == superimposed_atom_graph.num_nodes()
        assert concat_bond_embed.shape[0] == superimposed_atom_graph.num_edges()
        new_node_repr = self.superimposed_gnn.forward(superimposed_atom_graph, superimposed_atom_repr, concat_bond_embed)

        return new_node_repr


class HyperParams(Protocol):
    """Type hint for `self.hparams` in `ReactionDownstreamModel`."""
    encoder_params: dict[str, Any]
    out_size: int
    dropout_rate: float
    lr: float
    _logged_hparams: LoggedHyperParams

class ReactionDownstreamModel(GeoGNNLightningModule):
    """Downstream model for reaction-property prediction.

    Uses the node-representation output from `self.encoder: GeoGNNModel` to make
    `out_size` number of predictions.
    """

    @overload
    def __init__(self, *, encoder_params: dict[str, Any], out_size: int, dropout_rate: float, lr: float = 1e-4, _logged_hparams: LoggedHyperParams = {}) -> None:
        """
        Args:
            encoder_params (dict[str, Any]): Params to init a new `GeoGNNModel` \
                instance with.
            out_size (int): Output size (ie. number of predictions).
            dropout_rate (float): Rate for dropout layers.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file. Defaults to {}.
        """
    @overload
    def __init__(self, *, encoder: GeoGNNModel, out_size: int, dropout_rate: float, lr: float = 1e-4, _logged_hparams: LoggedHyperParams = {}) -> None:
        """
        Args:
            encoder (GeoGNNModel): GeoGNN encoder for the molecule graphs.
            out_size (int): Output size (ie. number of predictions).
            dropout_rate (float): Rate for dropout layers.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file. Defaults to {}.
        """
    def __init__(
        self,
        *,
        encoder: GeoGNNModel | None = None,
        encoder_params: dict[str, Any] | None = None,
        out_size: int,
        dropout_rate: float,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ) -> None:
        super().__init__(out_size, lr, _logged_hparams)
        assert (encoder != None and encoder_params == None) \
            or (encoder == None and encoder_params != None), \
            'Either `encoder` or `encoder_params` must be given, but not both. ' \
                + "I don't want to deal with the case where if both are given, " \
                + "and `encoder_params` doesn't match the params in `encoder`"

        self.encoder = encoder or GeoGNNModel(**encoder_params) # type: ignore
        encoder_params = {
            'embed_dim': self.encoder.embed_dim,
            'dropout_rate': self.encoder.dropout_rate,
            'num_of_layers': self.encoder.num_of_layers,
        }

        self.hparams: HyperParams
        self.save_hyperparameters(ignore=['encoder'])

        self.norm = nn.LayerNorm(self.encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = 2,
            in_size = self.encoder.embed_dim,
            hidden_size = self.encoder.embed_dim * 4,
            out_size = out_size,
            activation = nn.LeakyReLU(),
            dropout_rate = dropout_rate,
        )

    @override
    def forward(self, batched_atom_bond_graph: DGLGraph, batched_bond_angle_graph: DGLGraph, batched_superimposed_atom_graph: DGLGraph) -> Tensor:
        """
        Args:
            batched_atom_bond_graph (DGLGraph): Graph (or batched graph) of \
                molecules with atoms as nodes, bonds as edges.
            batched_bond_angle_graph (DGLGraph): Graph (or batched graph) of \
                molecules with bonds as nodes, bond-angles as edges.

        Returns:
            Tensor: Predicted values with size `(self.out_size, )`.
        """
        batched_node_repr, batched_edge_repr \
            = self.encoder.forward(batched_atom_bond_graph, batched_bond_angle_graph, pool_graph=False)

        pred_list: list[Tensor] = []
        for atom_bond_graph, bond_angle_graph, node_repr in split_batched_data(
            batched_atom_bond_graph = batched_atom_bond_graph,
            batched_bond_angle_graph = batched_bond_angle_graph,
            batched_node_repr = batched_node_repr,
        ):
            assert isinstance(atom_bond_graph, DGLGraph) \
                and isinstance(bond_angle_graph, DGLGraph) \
                and isinstance(node_repr, Tensor)
            reactant_node_repr, product_node_repr \
                = split_reactant_product_node_feat(node_repr, atom_bond_graph)

            diff_node_repr = product_node_repr - reactant_node_repr # shape (num_nodes, embed_dim)

            # Sum over the node dimension
            graph_repr = diff_node_repr.sum(dim=0)  # shape is now (embed_dim, )

            graph_repr = self.norm.forward(graph_repr)
            pred = self.mlp.forward(graph_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)
