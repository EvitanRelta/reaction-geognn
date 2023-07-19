from typing import Protocol, overload

import torch
from base_classes import GeoGNNLightningModule, LoggedHyperParams
from dgl import DGLGraph
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from geognn import GeoGNNModel, InnerGNN, Preprocessing
from geognn.layers import DropoutMLP, FeaturesEmbedding, FeaturesRBF
from torch import Tensor, nn
from typing_extensions import override

from .graph_utils import split_batched_data, split_reactant_product_node_feat


class AggregationGNN(nn.Module):
    """Layer that aggregates the encoder's output node-representation with the
    initial features to give a new node-representation."""

    def __init__(
        self,
        unconcat_embed_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        # Embeddings.
        self.bond_embedding = FeaturesEmbedding(Preprocessing.FEATURES['bond_feats'], unconcat_embed_dim)
        self.bond_rbf = FeaturesRBF(Preprocessing.RBF_PARAMS['bond'], unconcat_embed_dim)

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
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.
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
    """Type hint for `self.hparams` in `ProtoModel`."""
    embed_dim: int
    gnn_layers: int
    out_size: int
    dropout_rate: float
    lr: float
    _logged_hparams: LoggedHyperParams

class ProtoModel(GeoGNNLightningModule):
    @overload
    def __init__(self, *, embed_dim: int, gnn_layers: int, out_size: int, dropout_rate: float, lr: float = 1e-4, _logged_hparams: LoggedHyperParams = {}) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            gnn_layers (int): Number of GNN message-passing layers.
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
        embed_dim: int | None = None,
        gnn_layers: int | None = None,
        out_size: int,
        dropout_rate: float,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ) -> None:
        super().__init__(lr, _logged_hparams)

        if encoder:
            self.encoder = encoder
        else:
            assert (embed_dim != None) and (gnn_layers != None)
            self.encoder = GeoGNNModel(
                embed_dim = embed_dim,
                dropout_rate = dropout_rate,
                num_of_layers = gnn_layers,
            )

        self.hparams: HyperParams
        self.save_hyperparameters(ignore=['encoder', 'embed_dim', 'gnn_layers'])
        self.save_hyperparameters({
            '_geognn_encoder_hparams': {
                'embed_dim': self.encoder.embed_dim,
                'dropout_rate': self.encoder.dropout_rate,
                'num_of_layers': self.encoder.num_of_layers,
            }
        })

        # Dimension after concatenating reactant and diff node repr.
        concat_embed_dim = 2 * self.encoder.embed_dim
        self.norm = nn.LayerNorm(concat_embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = 2,
            in_size = concat_embed_dim,
            hidden_size = self.encoder.embed_dim * 4,
            out_size = out_size,
            activation = nn.LeakyReLU(),
            dropout_rate = dropout_rate,
        )

        # For aggregating node-repr with initial edge-feats, b4 pooling.
        self.aggregate_gnn = AggregationGNN(self.encoder.embed_dim, dropout_rate)
        self.graph_pool = GlobalAttentionPooling(nn.Linear(concat_embed_dim, 1))

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
        for atom_bond_graph, bond_angle_graph, superimposed_atom_graph, node_repr in split_batched_data(
            batched_atom_bond_graph = batched_atom_bond_graph,
            batched_bond_angle_graph = batched_bond_angle_graph,
            batched_superimposed_atom_graph = batched_superimposed_atom_graph,
            batched_node_repr = batched_node_repr,
        ):
            assert isinstance(atom_bond_graph, DGLGraph) \
                and isinstance(bond_angle_graph, DGLGraph) \
                and isinstance(superimposed_atom_graph, DGLGraph) \
                and isinstance(node_repr, Tensor)
            reactant_node_repr, product_node_repr \
                = split_reactant_product_node_feat(node_repr, atom_bond_graph)

            diff_node_repr = product_node_repr - reactant_node_repr # shape (num_nodes, embed_dim)
            concat_reactant_diff = torch.cat((reactant_node_repr, diff_node_repr), dim=1) # shape (num_nodes, 2 * embed_dim)

            assert concat_reactant_diff.shape[0] == superimposed_atom_graph.num_nodes()

            node_repr = self.aggregate_gnn.forward(superimposed_atom_graph, concat_reactant_diff)
            graph_repr = self.graph_pool.forward(superimposed_atom_graph, node_repr) # shape (1, 2 * embed_dim)
            assert isinstance(graph_repr, Tensor)
            graph_repr = graph_repr.squeeze(0) # shape (2 * embed_dim, )

            graph_repr = self.norm.forward(graph_repr)
            pred = self.mlp.forward(graph_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)
