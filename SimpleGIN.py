from torch import nn, Tensor
from torch_geometric import nn as gnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Union


class SimpleGIN(MessagePassing):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features.
    """
    def __init__(self, embed_dim: int):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(
        self,
        x: Union[Tensor, OptPairTensor], 
        edge_index: Adj, 
        edge_attr: OptTensor = None, 
        size: Size = None
    ) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:    
        return x_j + edge_attr
