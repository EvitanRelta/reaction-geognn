from typing import Protocol

import torch
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.layers import DropoutMLP
from lightning_utils import GeoGNNLightningModule, LoggedHyperParams
from torch import Tensor, nn
from typing_extensions import override

from .graph_utils import split_batched_data, split_reactant_product_node_feat


class HyperParams(Protocol):
    """Type hint for `self.hparams` in `ProtoModel`."""
    embed_dim: int
    gnn_layers: int
    out_size: int
    dropout_rate: float
    lr: float
    _logged_hparams: LoggedHyperParams

class ProtoModel(GeoGNNLightningModule):
    def __init__(
        self,
        embed_dim: int,
        gnn_layers: int,
        out_size: int,
        dropout_rate: float,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ) -> None:
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
        super().__init__(lr, _logged_hparams)
        self.hparams: HyperParams
        self.save_hyperparameters()

        self.compound_encoder = GeoGNNModel(
            embed_dim = embed_dim,
            dropout_rate = dropout_rate,
            num_of_layers = gnn_layers,
        )
        self.norm = nn.LayerNorm(self.compound_encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = 2,
            in_size = self.compound_encoder.embed_dim,
            hidden_size = self.compound_encoder.embed_dim * 4,
            out_size = out_size,
            activation = nn.LeakyReLU(),
            dropout_rate = dropout_rate,
        )

    @override
    def forward(self, batched_atom_bond_graph: DGLGraph, batched_bond_angle_graph: DGLGraph) -> Tensor:
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
            = self.compound_encoder.forward(batched_atom_bond_graph, batched_bond_angle_graph, pool_graph=False)

        pred_list: list[Tensor] = []
        for atom_bond_graph, node_repr in split_batched_data(
            batched_atom_bond_graph = batched_atom_bond_graph,
            batched_node_repr = batched_node_repr,
        ):
            reactant_node_repr, product_node_repr \
                = split_reactant_product_node_feat(node_repr, atom_bond_graph)

            diff_node_repr = product_node_repr - reactant_node_repr

            # Sum over the node dimension
            diff_node_repr = diff_node_repr.sum(dim=0)  # shape is now (embed_dim, )

            diff_node_repr = self.norm.forward(diff_node_repr)
            pred = self.mlp.forward(diff_node_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)
