from typing import Protocol, cast

import dgl
import lightning.pytorch as pl
import torch
import torchmetrics
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.layers import DropoutMLP
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor, nn
from torch.optim import Adam

from .data_module import BATCH_TUPLE, Wb97DataModule


class HyperParams(Protocol):
    """Type hint for `self.hparams` in `ProtoModel`."""
    out_size: int
    dropout_rate: float
    encoder_lr: float
    head_lr: float

class ProtoModel(pl.LightningModule):
    def __init__(
        self,
        compound_encoder: GeoGNNModel,
        out_size: int,
        dropout_rate: float,
        encoder_lr: float = 1e-3,
        head_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hparams: HyperParams
        self.save_hyperparameters(ignore=['compound_encoder'])

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(self.compound_encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = 2,
            in_size = self.compound_encoder.embed_dim,
            hidden_size = 128,
            out_size = out_size,
            activation = nn.LeakyReLU(),
            dropout_rate = dropout_rate,
        )
        # Needed as auto-optimization doesn't support multiple optimizers.
        self.automatic_optimization = False

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
        compound_encoder_params = list(self.compound_encoder.parameters())
        model_params = list(self.parameters())

        # Head-model params, excluding those in `self.compound_encoder`.
        # `head_params` is the difference in elements between `model_params` &
        # `compound_encoder_params`.
        is_in = lambda x, lst: any(element is x for element in lst)
        head_params = [p for p in model_params if not is_in(p, compound_encoder_params)]

        assert self.hparams.encoder_lr and self.hparams.head_lr
        encoder_optim = Adam(compound_encoder_params, lr=self.hparams.encoder_lr)
        head_optim = Adam(head_params, lr=self.hparams.head_lr)
        return encoder_optim, head_optim


    def training_step(self, batch: BATCH_TUPLE, batch_idx: int) -> Tensor:
        # Zeroing optimizers' gradients.
        encoder_optim, head_optim = cast(list[LightningOptimizer], self.optimizers())
        encoder_optim.zero_grad() # type: ignore
        head_optim.zero_grad() # type: ignore

        atom_bond_batch_graph, bond_angle_batch_graph, labels = batch
        pred = self.forward(atom_bond_batch_graph, bond_angle_batch_graph)
        loss = self.loss_fn(pred, labels)
        assert isinstance(loss, Tensor)

        # Log loss to the progress bar and logger
        datamodule = cast(Wb97DataModule, self.trainer.datamodule) # type: ignore
        rmse_loss = torch.sqrt(loss) * datamodule.scaler.fit_std.to(loss) # type: ignore
        self.log("train_loss", rmse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Manual backpropagation.
        self.manual_backward(loss)
        encoder_optim.step()
        head_optim.step()
        return loss

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
        datamodule = cast(Wb97DataModule, self.trainer.datamodule) # type: ignore
        pred = datamodule.scaler.inverse_transform(pred)
        labels = datamodule.scaler.inverse_transform(labels)

        loss = self.metric(pred, labels)
        assert isinstance(loss, Tensor)
        return loss
