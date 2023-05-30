from torch import nn, Tensor
from dgl import DGLGraph, function as fn
from typing import cast


class SimpleGIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer
    for undirected graphs, that accounts for with edge features.
    """
    def __init__(self, hidden_size: int) -> None:
        super(SimpleGIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
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
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            graph.update_all(
                message_func=fn.u_add_e('h', 'h', 'm'),
                reduce_func=fn.sum("m", "h_out"),
            )
            output_node_feats = graph.ndata["h_out"]
            return self.mlp.forward(output_node_feats)
