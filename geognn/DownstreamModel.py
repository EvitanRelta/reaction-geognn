from typing import Literal

from dgl import DGLGraph
from torch import Tensor, nn

from .GeoGNN import GeoGNNModel
from .layers import DropoutMLP


class DownstreamModel(nn.Module):
    """
    Model that uses the graph-representation output from
    `self.compound_encoder: GeoGNNModel` to make `out_size` number of predictions.

    This is a PyTorch + DGL equivalent of GeoGNN's `DownstreamModel`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/src/model.py#L27-L61
    """

    def __init__(
        self,
        compound_encoder: GeoGNNModel,
        task_type: Literal['classification', 'regression'],
        out_size: int,
        num_of_mlp_layers: int,
        mlp_hidden_size: int = 128,
        activation: nn.Module = nn.LeakyReLU(),
        dropout_rate: float = 0.2,
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
            compound_encoder (GeoGNNModel): GeoGNN encoder for the molecule graphs.
            task_type (Literal['classification', 'regression']): Whether to \
                perform a classification or regression.
            out_size (int): Size of output tensor.
            num_of_mlp_layers (int): Number of layers in the dropout MLP.
            mlp_hidden_size (int, optional): Hidden size of dropout MLP. Defaults to 128.
            activation (nn.Module, optional): Activation layer to use. Defaults to `nn.LeakyReLU()`.
            dropout_rate (float, optional): Dropout rate of the dropout MLP. Defaults to 0.2.
        """
        super().__init__()
        self.task_type = task_type
        self.out_size = out_size

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = num_of_mlp_layers,
            in_size = compound_encoder.embed_dim,
            hidden_size = mlp_hidden_size,
            out_size = out_size,
            activation = activation,
            dropout_rate = dropout_rate,
        )
        if self.task_type == 'classification':
            self.out_act = nn.Sigmoid()

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
        node_repr, edge_repr, graph_repr \
            = self.compound_encoder.forward(atom_bond_graph, bond_angle_graph)
        graph_repr = self.norm.forward(graph_repr)
        pred = self.mlp.forward(graph_repr)

        if self.task_type == 'classification':
            return self.out_act.forward(pred)
        return pred
