from __future__ import annotations

import numpy as np

from ..flags import _NO_GRAD
from ..operations import Mul
from ..types import (
    _float16, _float32, _float64,
    _TensorArray,
    ensure_type, ensure_shape
)

class Tensor:
    def __init__(self, shape, dtype = _float32, requires_grad: bool = False) -> None:
        self.__dtype = dtype
        self.__values = np.zeros(shape, dtype = dtype)
        self.__grads = np.zeros(shape, dtype = dtype)
        self.__requires_grad = requires_grad
        self.__children: list[Tensor] = []

    @classmethod
    def zeros(cls, shape, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, dtype, requires_grad)
        tensor.values = np.zeros(shape, dtype=dtype)
        return tensor
    
    @classmethod
    def ones(cls, shape, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, dtype, requires_grad)
        tensor.values = np.ones(shape, dtype=dtype)
        return tensor
    
    @classmethod
    def full(cls, shape, value, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, dtype, requires_grad)
        tensor.values = np.full(shape, value, dtype=dtype)
        return tensor
    
    @classmethod
    def randn(cls, shape, mean = 0, stddev = 1, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, dtype, requires_grad)
        tensor.values = np.random.normal(mean, stddev, shape) # will be casted by ensure_type
        return tensor
    
    @classmethod
    def random(cls, shape, min = 0, max = 1, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, dtype, requires_grad)
        tensor.values = np.random.rand(*shape) * (max - min) + min
        return tensor
    
    @classmethod
    def from_values(cls, values: _TensorArray, dtype = _float32, requires_grad: bool = False) -> Tensor:
        tensor = cls(values.shape, dtype, requires_grad)
        tensor.values = values 
        return tensor

    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def shape(self):
        return self.__values.shape # we could use grad but it's the same in the end

    @property
    def values(self) -> _TensorArray:
        return self.__values
    
    @values.setter
    @ensure_type
    @ensure_shape
    def values(self, other) -> None:
        self.__values = other

    @property
    def grads(self) -> _TensorArray:
        return self.__grads
    
    @ensure_type
    def accumulate_grads(self, grad_updates: _TensorArray) -> None:
        if self.__requires_grad and not _NO_GRAD:
            self.__grads += grad_updates

    def add_children(self, *children) -> None:
        for child in children:
            self.__children.append(child)

    # OPERATIONS
    def __mul__(self, other: Tensor) -> Tensor:
        result = Mul.forward(self.values, other.values)
        tensor = Tensor.from_values(result, dtype=self.dtype, requires_grad=self.__requires_grad)
        return tensor

