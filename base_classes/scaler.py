"""Scaler class for transforming dataset labels."""

import torch
from torch import Tensor


class StandardizeScaler:
    """Scaler to standardize Tensors against the mean/std of the fitted Tensor."""
    def __init__(self):
        self._fit_mean: Tensor | None = None
        self._fit_std: Tensor | None = None

    @property
    def has_fitted(self) -> bool:
        return (self._fit_mean is not None) \
            and (self._fit_std is not None)

    @property
    def fit_mean(self) -> Tensor:
        if self._fit_mean == None:
            raise RuntimeError('Scaler has not not been fitted yet.')
        return self._fit_mean
    @fit_mean.setter
    def fit_mean(self, value) -> None:
        self._fit_mean = value

    @property
    def fit_std(self) -> Tensor:
        if self._fit_std == None:
            raise RuntimeError('Scaler has not not been fitted yet.')
        return self._fit_std
    @fit_std.setter
    def fit_std(self, value) -> None:
        self._fit_std = value

    def fit(self, x: Tensor) -> None:
        self.fit_mean = torch.mean(x, dim=0)
        self.fit_std = torch.std(x, dim=0)

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
