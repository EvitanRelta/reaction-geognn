import torch
from torch import nn, Tensor


class FixedRBF(nn.Module):
    """
    Radial Basis Function (RBF) neural network layer that doesn't have any
    learnable parameters (ie. "fixed"). It simply transforms the features.
    
    This is a Pytorch equivalent of GeoGNN's `RBF`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/basic_block.py#L71
    """
    def __init__(self, centers: Tensor, gamma: float) -> None:
        """
        Args:
            centers (Tensor): 1D tensor of all the RBF centers.
            gamma (Tensor): Hyperparameter for controlling the spread of the RBF's Gaussian basis function.
        """
        super(FixedRBF, self).__init__()
        self.centers = torch.reshape(centers.float(), [1, -1])
        """2D row-tensor containing the RBF centers. Has shape `(1, num_of_centers)`."""
        self.gamma = gamma
        """Hyperparameter for controlling the spread of the RBF's Gaussian basis function."""
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Tensor with additional dimension of length `num_of_centers` \
                (eg. if `x` is shape `(n, m)`, output shape will be `(n, m, num_of_centers)`).
        """
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))
