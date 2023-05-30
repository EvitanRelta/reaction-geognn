import torch
from torch import Tensor, nn
from Utils import FeatureName, RBFCenters, RBFGamma
from typing import cast
from FixedRBF import FixedRBF


class FeaturesRBF(nn.Module):
    """
    Transforms multiple features (eg. node/edge features) into a tensor of size
    `(num_elements, self.output_dim)`.
    
    It works by first converting each feature into a tensor of size
    `(num_elements, self.output_dim)`, then summing all the features' tensors
    into one tensor.
    """

    def __init__(
        self, 
        rbf_param_dict: dict[FeatureName, tuple[RBFCenters, RBFGamma]], 
        output_dim: int
    ):
        """
        Args:
            rbf_param_dict (dict[FeatureName, tuple[RBFCenters, RBFGamma]]): \
                Dict containing the RBF parameters for every feature.
            output_dim (int): The output embedding dimension.
        """
        super(FeaturesRBF, self).__init__()

        self.output_dim = output_dim

        # Temp. type for fixing type hints.
        _ZippedType = tuple[tuple[FeatureName, ...], tuple[tuple[RBFCenters, RBFGamma], ...]]
        self.feat_names, self.rbf_params = cast(_ZippedType, zip(*rbf_param_dict.items()))

        self.module_list = nn.ModuleList()
        for centers, gamma in self.rbf_params:
            layer = nn.Sequential(
                FixedRBF(centers, gamma),
                nn.Linear(len(centers), output_dim)
            )
            self.module_list.append(layer)

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

        for i, feat_name in enumerate(self.feat_names):
            layer: nn.Sequential = self.module_list[i]
            feat = feat_tensor_dict[feat_name]
            layer_output = layer.forward(feat)
            output_embed += layer_output

        return output_embed
