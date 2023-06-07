from torch import nn, Tensor
from dgl import DGLGraph, function as fn
from typing import cast


class SimpleGIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer
    for undirected graphs, that accounts for with edge features.
    """
    def __init__(self, mlp_in_size: int, mlp_hidden_size: int, mlp_out_size: int) -> None:
        """
        - `MLP` - Multilayer Perceptron
        Args:
            mlp_in_size (int): Input size of the MLP.
            mlp_hidden_size (int): Hidden layer's size of the MLP.
            mlp_out_size (int): Output size of the MLP.
        """
        super(SimpleGIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_out_size)
        )

    def reset_parameters(self) -> None:
        """Resets the weights of `self.mlp`."""
        cast(nn.Linear, self.mlp[0]).reset_parameters()
        cast(nn.Linear, self.mlp[2]).reset_parameters()

    def forward(self, graph: DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tensor:
        """
        Args:
            graph (DGLGraph): The input graph.
            node_feat (Tensor): The input node features.
            edge_feats (Tensor): The input edge features.
        """
        with graph.local_scope():
            graph.ndata['h_n'] = node_feats
            graph.edata['h_e'] = edge_feats
            graph.update_all(
                message_func=fn.u_add_e('h_n', 'h_e', 'm'),
                reduce_func=fn.sum('m', 'h_out'),
            )
            output_node_feats = graph.ndata['h_out']
            return self.mlp.forward(output_node_feats)
