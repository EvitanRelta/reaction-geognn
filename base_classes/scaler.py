"""Scaler class for transforming dataset labels."""

import torch
from torch import Tensor, nn


# Inherit from `nn.Module` so that the mean/std tensors are save/loaded along
# with the model's state_dict.
class StandardizeScaler(nn.Module):
    """Scaler to standardize Tensors against the mean/std of the fitted Tensor."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

        self._has_fitted: Tensor
        self._fit_mean: Tensor
        self._fit_std: Tensor

        # Register as buffers, so that they'll be saved/loaded in state_dict.
        self.register_buffer('_has_fitted', torch.BoolTensor([False]))
        self.register_buffer('_fit_mean', torch.empty(self.size))
        self.register_buffer('_fit_std', torch.empty(self.size))


    @property
    def has_fitted(self) -> bool:
        return bool(self._has_fitted.item())
    @has_fitted.setter
    def has_fitted(self, value) -> None:
        assert isinstance(value, bool)
        self._has_fitted[0] = value


    @property
    def fit_mean(self) -> Tensor:
        if not self.has_fitted:
            raise RuntimeError('Scaler has not not been fitted yet.')
        return self._fit_mean
    @fit_mean.setter
    def fit_mean(self, value) -> None:
        self._fit_mean = value


    @property
    def fit_std(self) -> Tensor:
        if not self.has_fitted:
            raise RuntimeError('Scaler has not not been fitted yet.')
        return self._fit_std
    @fit_std.setter
    def fit_std(self, value) -> None:
        self._fit_std = value


    def fit(self, x: Tensor) -> None:
        assert torch.is_floating_point(x) and x.dim() == 2, \
            f"Expected a 2D float tensor, got `dtype={x.dtype}` with `dim={x.dim()}`."
        assert x.shape[1] == self.size, \
            f"Expected a tensor of size `(N, {self.size})`, got size `(N, {x.shape[1]})`." \
            + f"\nIf tensor is intended to be of size `(N, {x.shape[1]})`, then initialise scaler with `StandardizeScaler(size={x.shape[1]})` instead."
        self.fit_mean = torch.mean(x, dim=0)
        self.fit_std = torch.std(x, dim=0)
        self.has_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        self._move_to_device(x)
        return (x - self.fit_mean) / self.fit_std

    def inverse_transform(self, x: Tensor) -> Tensor:
        self._move_to_device(x)
        return x * self.fit_std + self.fit_mean

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)

    def _move_to_device(self, x: Tensor) -> None:
        self.fit_mean = self.fit_mean.to(x)
        self.fit_std = self.fit_std.to(x)
