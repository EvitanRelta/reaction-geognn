import torch
from torch import Tensor
from typing import Optional
from torch_geometric.nn.pool import global_add_pool


class SqrtGraphNorm(torch.nn.Module):
    """
    Applies graph normalization, where each node features is divided by 
    sqrt(num_nodes) for each graph.
    
    This is a Pytorch equivalent of GeoGNN's `GraphNorm`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/dev/pahelix/networks/gnn_block.py#L26
    
    Adapted from:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/norm/graph_norm.html#GraphNorm
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): The source tensor.
            batch (Optional[Tensor], optional): The batch vector, which assigns
                each element to a specific example. (default: `None`)

        Returns:
            Tensor: _description_
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        num_nodes = global_add_pool(torch.ones_like(x), batch)
        norm_factor = num_nodes.sqrt().index_select(0, batch).clamp(min=1)
        return x / norm_factor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
