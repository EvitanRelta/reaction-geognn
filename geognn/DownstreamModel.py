from typing import Literal

from base_classes import GeoGNNLightningModule, LoggedHyperParams
from dgl import DGLGraph
from torch import Tensor, nn
from typing_extensions import override

from .GeoGNN import GeoGNNModel
from .layers import DropoutMLP


class DownstreamModel(GeoGNNLightningModule):
    """
    Model that uses the graph-representation output from
    `self.encoder: GeoGNNModel` to make `out_size` number of predictions.

    This is a PyTorch + DGL equivalent of GeoGNN's `DownstreamModel`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/src/model.py#L27-L61
    """

    def __init__(
        self,
        encoder: GeoGNNModel,
        task_type: Literal['classification', 'regression'],
        out_size: int,
        num_of_mlp_layers: int,
        mlp_hidden_size: int = 128,
        activation: nn.Module = nn.LeakyReLU(),
        dropout_rate: float = 0.2,
        lr: float = 1e-4,
        _logged_hparams: LoggedHyperParams = {},
    ):
        """
        Default values for `hidden_size`, `activation` and `dropout_rate` are
        based on GeoGNN's `down_mlp2.json` / `down_mlp3.json` config:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/down_mlp2.json
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/down_mlp3.json

        where `down_mlp2.json` and `down_mlp3.json` are used by the GeoGNN's
        finetuning scripts:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_class.sh#L37
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/scripts/finetune_regr.sh#L37

        Both of GeoGNN's finetuning scripts tests both `num_of_layers = 2` and
        `num_of_layers = 3`.

        Args:
            encoder (GeoGNNModel): GeoGNN encoder for the molecule graphs.
            task_type (Literal['classification', 'regression']): Whether to \
                perform a classification or regression.
            out_size (int): Size of output tensor.
            num_of_mlp_layers (int): Number of layers in the dropout MLP.
            mlp_hidden_size (int, optional): Hidden size of dropout MLP. Defaults to 128.
            activation (nn.Module, optional): Activation layer to use. Defaults to `nn.LeakyReLU()`.
            dropout_rate (float, optional): Dropout rate of the dropout MLP. Defaults to 0.2.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            _logged_hparams (LoggedHyperParams, optional): Hyperparameters that's \
                not used by the model, but is logged in the lightning-log's \
                `hparams.yaml` file.. Defaults to {}.
        """
        super().__init__(lr, _logged_hparams)
        self.save_hyperparameters(ignore=['encoder', 'activation'])
        self.save_hyperparameters({
            '_geognn_encoder_hparams': {
                'embed_dim': encoder.embed_dim,
                'dropout_rate': encoder.dropout_rate,
                'num_of_layers': encoder.num_of_layers,
            }
        })
        self.task_type = task_type
        self.out_size = out_size

        self.encoder = encoder
        self.norm = nn.LayerNorm(encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = num_of_mlp_layers,
            in_size = encoder.embed_dim,
            hidden_size = mlp_hidden_size,
            out_size = out_size,
            activation = activation,
            dropout_rate = dropout_rate,
        )
        if self.task_type == 'classification':
            self.out_act = nn.Sigmoid()

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
        node_repr, edge_repr, graph_repr \
            = self.encoder.forward(batched_atom_bond_graph, batched_bond_angle_graph)
        graph_repr = self.norm.forward(graph_repr)
        pred = self.mlp.forward(graph_repr)

        if self.task_type == 'classification':
            return self.out_act.forward(pred)
        return pred
