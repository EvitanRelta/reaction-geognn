from typing import cast

import torch
from torch import Tensor, nn

from ..Preprocessing import Feature, FeatureName


class FeaturesEmbedding(nn.Module):
    """
    Converts multiple features (eg. node/edge features) into an embedding of
    size `(num_elements, self.embed_dim)`.

    It works by first converting each feature into an embedding of size
    `(num_elements, self.embed_dim)`, then summing all the features' embeddings
    into one embedding.

    This is a generalised PyTorch equivalent of GeoGNN's `AtomEmbedding`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L28-L52

    and `BondEmbedding`:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/networks/compound_encoder.py#L94-L118
    """

    def __init__(
        self,
        feat_dict: dict[FeatureName, Feature],
        embed_dim: int,
        feat_padding: int = 5,
    ):
        """
        Args:
            feat_dict (dict[FeatureName, Feature]): Dict containing \
                the info for every feature.
            embed_dim (int): The output embedding dimension.
            feat_padding (int): The extra num of embeddings added to \
                `num_embeddings` in `nn.Embedding` to act as padding.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dict = nn.ModuleDict()
        for feat_name, feat in feat_dict.items():
            input_dim = len(feat.possible_values) + feat_padding
            self.embed_dict[feat_name] = nn.Embedding(input_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the weights of all the `nn.Embedding` submodules by reinitializing
        each of their weights using Xavier Uniform initialisation.
        """
        for _, embedding in self.embed_dict.items():
            nn.init.xavier_uniform_(cast(nn.Embedding, embedding).weight)

    def forward(self, feat_tensor_dict: dict[FeatureName, Tensor]) -> Tensor:
        """
        Args:
            feat_tensor_dict (dict[FeatureName, Tensor]): Dictionary of \
                features, where the keys are the features' names and the values \
                are the features' `Tensor` values (eg. from `DGLGraph.ndata`).

        Returns:
            Tensor: Embedding of size `(num_elements, self.embed_dim)`.
        """
        feat_values = list(feat_tensor_dict.values())
        num_of_elements = len(feat_values[0])
        device = next(self.parameters()).device
        output_embed = torch.zeros(num_of_elements, self.embed_dim, dtype=torch.float32, device=device)

        for feat_name, tensor in feat_tensor_dict.items():
            if feat_name not in self.embed_dict:
                continue
            embedding_layer: nn.Embedding = self.embed_dict[feat_name]
            layer_output = embedding_layer.forward(tensor)
            output_embed += layer_output

        return output_embed
