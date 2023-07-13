from abc import ABC, abstractmethod
from typing import Any, TypedDict

import lightning.pytorch as pl
import torch
import torchmetrics
from dgl import DGLGraph
from geognn_base_classes import GeoGNNBatch
from torch import Tensor
from torch.optim import Adam
from typing_extensions import NotRequired

from .scaler import StandardizeScaler


class LoggedHyperParams(TypedDict):
    """Hyperparameters that's not used by the model, but is logged in the
    lightning-log's `hparams.yaml` file.
    """
    batch_size: NotRequired[int]
    dataset_size: NotRequired[int]
    notes: NotRequired[str]

class GeoGNNLightningModule(ABC, pl.LightningModule):
    """Abstract base class for PyTorch-Lightning data-modules for GeoGNN
    downstream models.

    ### Implements:
    - `Adam` optimizer with `self.lr` specified by the constructor's `lr` arg.
    - MSE as training loss function.
    - RMSE as test/validation metric function.
    - Logging of the below losses/metrics to lightning-log's `metrics.csv` file"
      - `"train_raw_std_mse_loss"` - Mean of all standardized training MSE batch
        losses in an epoch.
      - `"train_loss"` - Unstandardized training RMSE computed using all prediction
        values (ie. not using the batch losses) at the end of an epoch.
      - `"val_loss"` - Unstandardized validation RMSE computed using all prediction
        values (ie. not using the batch losses) at the end of an epoch.
      - `"test_loss"` - Unstandardized test RMSE computed using all prediction
        values (ie. not using the batch losses) at the end of an epoch.
    - Logging of hyper-params that isn't used in the model (eg. batch size,
      developer notes) to lightning-log's `hparams.yaml` file.

    ### Requires the below abstract methods to be implemented:
    ```python
    @abstractmethod
    def forward(self, batched_atom_bond_graph: DGLGraph, batched_bond_angle_graph: DGLGraph) -> Tensor:
    ```
    """

    @abstractmethod
    def forward(self, batched_atom_bond_graph: DGLGraph, batched_bond_angle_graph: DGLGraph) -> Tensor:
        """
        Args:
            batched_atom_bond_graph (DGLGraph): Graph (or batched graph) of \
                molecules with atoms as nodes, bonds as edges.
            batched_bond_angle_graph (DGLGraph): Graph (or batched graph) of \
                molecules with bonds as nodes, bond-angles as edges.

        Returns:
            Tensor: Predicted values with size `(num_tasks, )`, where `num_tasks` \
                is the number of tasks given in the `labels` of a `GeoGNNBatch`.
        """

    def __init__(
        self,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ) -> None:
        """
        Args:
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file.. Defaults to {}.
        """
        super().__init__()
        self.save_hyperparameters('_logged_hparams')

        self.lr = lr
        self.scaler: StandardizeScaler = StandardizeScaler()

        # Loss/Metric functions.
        self.loss_fn = torchmetrics.MeanSquaredError() # MSE
        self.metric = torchmetrics.MeanSquaredError(squared=False) # RMSE

        # For storing pred/label values.
        self._train_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.
        self._test_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.
        self._val_step_values: list[tuple[Tensor, Tensor]] = [] # (pred, label) for each batch/step.


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
