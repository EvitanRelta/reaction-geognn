from typing import Any, Protocol

import lightning.pytorch as pl
import torch
import torchmetrics
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.datasets import GeoGNNBatch
from geognn.layers import DropoutMLP
from lightning_utils import StandardizeScaler
from torch import Tensor, nn
from torch.optim import Adam

from .graph_utils import split_batched_data, split_reactant_product_node_feat


class HyperParams(Protocol):
    """Type hint for `self.hparams` in `ProtoModel`."""
    embed_dim: int
    gnn_layers: int
    out_size: int
    dropout_rate: float
    lr: float
    _batch_size: int | None
    _dataset_size: int | None

class ProtoModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        gnn_layers: int,
        out_size: int,
        dropout_rate: float,
        lr: float = 1e-3,
        _batch_size: int | None = None,
        _dataset_size: int | None = None,
        _notes: str | None = None,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            gnn_layers (int): Number of GNN message-passing layers.
            out_size (int): Output size (ie. number of predictions).
            dropout_rate (float): Rate for dropout layers.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            _batch_size (int | None, optional): Size of batches used during training. \
                Not used in the model, but just to log in the `hparams.yaml`. \
                Defaults to None.
            _dataset_size (int | None, optional): Size of dataset used during training. \
                Not used in the model, but just to log in the `hparams.yaml`. \
                Defaults to None.
            _notes (str | None, optional): Developer notes for the model. \
                Not used in the model, but just to log in the `hparams.yaml`. \
                Defaults to None.
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

        # For storing pred/label values.
        self._train_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.
        self._test_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.
        self._val_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.

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
        for graph, node_repr in split_batched_data(
            batched_atom_bond_graph = batched_atom_bond_graph,
            batched_node_repr = batched_node_repr,
        ):
            reactant_node_repr, product_node_repr \
                = split_reactant_product_node_feat(node_repr, graph)

            diff_node_repr = product_node_repr - reactant_node_repr

            # Sum over the node dimension
            diff_node_repr = diff_node_repr.sum(dim=0)  # shape is now (embed_dim, )

            diff_node_repr = self.norm.forward(diff_node_repr)
            pred = self.mlp.forward(diff_node_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)


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


    def training_step(self, batch: GeoGNNBatch, batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        self._train_step_values.append((pred, labels))
        loss = self.loss_fn(pred, labels)

        # Log raw unstandardized MSE loss to the progress bar and logger.
        self.log("train_raw_std_mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch: GeoGNNBatch | tuple[DGLGraph, DGLGraph], batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, *_ = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        pred = self.scaler.inverse_transform(pred)
        return pred

    def validation_step(self, batch: GeoGNNBatch, batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        self._val_step_values.append((pred, labels))

        # Unstandardize values before computing loss.
        pred = self.scaler.inverse_transform(pred)
        labels = self.scaler.inverse_transform(labels)

        loss = self.metric(pred, labels)
        return loss

    def test_step(self, batch: GeoGNNBatch, batch_idx: int) -> Tensor:
        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        self._test_step_values.append((pred, labels))

        # Unstandardize values before computing loss.
        pred = self.scaler.inverse_transform(pred)
        labels = self.scaler.inverse_transform(labels)

        loss = self.metric(pred, labels)
        return loss


    def on_train_epoch_end(self) -> None:
        loss = self._compute_full_dataset_unstd_metric(self._train_step_values)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self._train_step_values = []

    def on_validation_epoch_end(self) -> None:
        loss = self._compute_full_dataset_unstd_metric(self._val_step_values)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self._val_step_values = []

    def on_test_epoch_end(self) -> None:
        loss = self._compute_full_dataset_unstd_metric(self._test_step_values)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self._test_step_values = []

    def _compute_full_dataset_unstd_metric(self, step_values: list[tuple[Tensor, Tensor]]) -> Tensor:
        # Unzip the predictions and labels.
        pred_list, labels_list = zip(*step_values)

        # Concatenate prediction/labels tensors.
        all_pred = torch.cat(pred_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        # Unstandardize values.
        all_unstd_pred = self.scaler.inverse_transform(all_pred)
        all_unstd_labels = self.scaler.inverse_transform(all_labels)

        full_dataset_unstd_loss = self.metric(all_unstd_pred, all_unstd_labels)
        return full_dataset_unstd_loss