from typing import Any, Protocol

import dgl
import lightning.pytorch as pl
import torch
import torchmetrics
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.layers import DropoutMLP
from torch import Tensor, nn
from torch.optim import Adam

from .data_module import BATCH_TUPLE, StandardizeScaler


class HyperParams(Protocol):
    """Type hint for `self.hparams` in `ProtoModel`."""
    embed_dim: int
    gnn_layers: int
    out_size: int
    dropout_rate: float
    lr: float
    _batch_size: int | None

class ProtoModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        gnn_layers: int,
        out_size: int,
        dropout_rate: float,
        lr: float = 1e-3,
        _batch_size: int | None = None,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            gnn_layers (int): Number of GNN message-passing layers.
            out_size (int): Output size (ie. number of predictions).
            dropout_rate (float): Rate for dropout layers.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            _batch_size (int | None, optional): Batch size used during training. \
                Not used in the model, but just to log the batch-size in the \
                `hparams.yaml`. Defaults to None.
        """
        super().__init__()
        self.hparams: HyperParams
        self.save_hyperparameters()

        self.lr = lr
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
        self.scaler: StandardizeScaler = StandardizeScaler()

        # Loss/Metric functions.
        self.loss_fn = torchmetrics.MeanSquaredError() # MSE
        self.metric = torchmetrics.MeanSquaredError(squared=False) # RMSE

    def forward(self, atom_bond_graph: DGLGraph, bond_angle_graph: DGLGraph) -> Tensor:
        """
        Args:
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.

        Returns:
            Tensor: Predicted values with size `(self.out_size, )`.
        """
        batched_node_repr, batched_edge_repr, batched_graph_repr \
            = self.compound_encoder.forward(atom_bond_graph, bond_angle_graph)


        pred_list: list[Tensor] = []
        for node_repr, graph in ProtoModel._split_batched_data(batched_node_repr, atom_bond_graph):
            reactant_node_repr, product_node_repr \
                = ProtoModel._split_reactant_product_nodes(node_repr, graph)

            diff_node_repr = product_node_repr - reactant_node_repr

            # Sum over the node dimension
            diff_node_repr = diff_node_repr.sum(dim=0)  # shape is now (embed_dim, )

            diff_node_repr = self.norm.forward(diff_node_repr)
            pred = self.mlp.forward(diff_node_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)

    @staticmethod
    def _split_batched_data(
        batched_node_repr: Tensor,
        batched_atom_bond_graph: DGLGraph,
    ) -> list[tuple[Tensor, DGLGraph]]:
        output: list[tuple[Tensor, DGLGraph]] = []
        start_index = 0
        for graph in dgl.unbatch(batched_atom_bond_graph):
            num_nodes = graph.number_of_nodes()
            node_repr = batched_node_repr[start_index : start_index + num_nodes]
            start_index += num_nodes
            output.append((node_repr, graph))
        return output

    @staticmethod
    def _split_reactant_product_nodes(
        node_repr: Tensor,
        atom_bond_graph: DGLGraph,
    ) -> tuple[Tensor, Tensor]:
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


    # ==========================================================================
    #                        Training-related methods
    # ==========================================================================
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def on_fit_start(self) -> None:
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'scaler'): # type: ignore
            self.scaler = self.trainer.datamodule.scaler # type: ignore
            assert isinstance(self.scaler, StandardizeScaler)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint['scaler'] = self.scaler

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.scaler = checkpoint['scaler']

    def training_step(self, batch: BATCH_TUPLE, batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        mse = self.loss_fn(pred, labels)
        assert isinstance(mse, Tensor)

        # Unstandardize loss for logging.
        std = self.scaler.fit_std.to(mse) # type: ignore
        assert isinstance(std, Tensor)
        unstandardized_mse = mse * (std ** 2)

        # Log loss to the progress bar and logger
        self.log("train_unstandardized_mse_loss", unstandardized_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return mse

    def predict_step(self, batch: BATCH_TUPLE | tuple[DGLGraph, DGLGraph], batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, *_ = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        pred = self.scaler.inverse_transform(pred)
        return pred

    def validation_step(self, batch: BATCH_TUPLE, batch_idx: int) -> Tensor:
        loss = self._metric_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: BATCH_TUPLE, batch_idx: int) -> Tensor:
        loss = self._metric_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _metric_step(self, batch: BATCH_TUPLE, batch_idx: int) -> Tensor:
        """Shared step function for both validation/test steps."""
        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)

        # Unstandardize values before computing loss.
        pred = self.scaler.inverse_transform(pred)
        labels = self.scaler.inverse_transform(labels)

        loss = self.metric(pred, labels)
        assert isinstance(loss, Tensor)
        return loss
