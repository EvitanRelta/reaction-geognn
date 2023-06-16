import torch
from torch import Tensor, nn
from ..Preprocessing import FeatureName, RBFCenters, RBFGamma
from typing import cast
from .FixedRBF import FixedRBF


class FeaturesRBF(nn.Module):
    """
    Transforms multiple features (eg. node/edge features) into a tensor of size
    `(num_elements, self.output_dim)`.

    It works by first converting each feature into a tensor of size
    `(num_elements, self.output_dim)`, then summing all the features' tensors
    into one tensor.

    This is a generalised PyTorch equivalent of GeoGNN's `BondFloatRBF` and
    `BondAngleFloatRBF`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L121-L192
    """

    def __init__(
        self,
        rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]],
        output_dim: int,
    ):
        """
        Args:
            rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]]): \
                Dict containing the RBF parameters for every feature.
            output_dim (int): The output embedding dimension.
        """
        super().__init__()
        self.output_dim = output_dim
        self.module_dict = nn.ModuleDict()
        for feat_name, (centers, gamma) in rbf_param_dict.items():
            layer = nn.Sequential(
                FixedRBF(centers, gamma),
                nn.Linear(len(centers), output_dim),
            )
            self.module_dict[feat_name] = layer

    def reset_parameters(self) -> None:
        """
        Resets the weights of all the `nn.Linear` submodules by calling
        `nn.Linear.reset_parameters()` on each of them.
        """
        for _, sequential in self.module_dict.items():
            cast(nn.Linear, sequential[1]).reset_parameters()

    def forward(self, feat_tensor_dict: dict[FeatureName, Tensor]) -> Tensor:
        """
        Args:
            feat_tensor_dict (dict[FeatureName, Tensor]): Dictionary of \
                features, where the keys are the features' names and the values \
                are the features' `Tensor` values (eg. from `DGLGraph.ndata`).

        Returns:
            Tensor: Embedding of size `(num_elements, self.output_dim)`.
        """
        feat_values = list(feat_tensor_dict.values())
        num_of_elements = len(feat_values[0])
        device = next(self.parameters()).device
        output_embed = torch.zeros(num_of_elements, self.output_dim, dtype=torch.float32, device=device)

        for feat_name, tensor in feat_tensor_dict.items():
            if feat_name not in self.module_dict:
                continue
            layer: nn.Sequential = self.module_dict[feat_name]
            layer_output = cast(Tensor, layer.forward(tensor))
            output_embed += layer_output

        return output_embed
