from typing import Any
from collections.abc import Callable
from .layer import Layer
from .. import xp
from ..tensor import Tensor
from ..math import dot
from ..types import auto_convert_to_cupy, _TensorArray

class Reshape(Layer):
    def __init__(self, target_shape) -> None:
        self._target_shape = target_shape

    def __call__(self, x: Tensor) -> Tensor | Any:
        if len(x.shape) < 2:
            raise Exception("Reshape must have minimum two dimensions (batch_size, *dims)") # TODO change with a shape error
        if xp.prod(x.shape[1:]) != xp.prod(self._target_shape):
            raise Exception(f"Input shape and target shape doesn't match : {x.shape}, {self._target_shape}") # TODO change with a shape error
        return x.reshape((x.shape[0], *self._target_shape))

    @property
    def parameters(self) -> list[Tensor]:
        return [] # empty list because no parameters

    @parameters.setter
    @auto_convert_to_cupy
    def parameters(self, params: list[Tensor] | list[xp.ndarray]) -> None:
        raise ValueError("Cannot assign any weight to a layer that has no learnable parameters")

class Flatten(Layer):
    def __init__(self, target_shape) -> None:
        self._target_shape = target_shape

    def __call__(self, x: Tensor) -> Tensor | Any:
        if len(x.shape) < 2:
            raise Exception("Reshape must have minimum two dimensions (batch_size, *dims)") # TODO change with a shape error
        return x.reshape((x.shape[0], xp.prod(x.shape[1:])))

    @property
    def parameters(self) -> list[Tensor]:
        return [] # empty list because no parameters

    @parameters.setter
    @auto_convert_to_cupy
    def parameters(self, params: list[Tensor] | list[xp.ndarray]) -> None:
        raise ValueError("Cannot assign any weight to a layer that has no learnable parameters")

