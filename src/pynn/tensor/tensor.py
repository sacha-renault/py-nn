from __future__ import annotations

import numpy as np

from ..flags import Flags
from ..operations import (Operation, Multiplication, Addition)
from ..types import (
    _float16, _float32, _float64,
    _TensorArray,
    ensure_type, ensure_shape
)

class Tensor:
    def __init__(self, shape, requires_grad: bool = False) -> None:
        self.__values = np.zeros(shape, dtype = Flags.global_type())
        self.__grads = np.zeros(shape, dtype = Flags.global_type())
        self.__requires_grad = requires_grad
        self.__children: list[Tensor] = []
        self.__op: Operation = None

    @classmethod
    def zeros(cls, shape, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = np.zeros(shape, dtype = Flags.global_type())
        print(np.may_share_memory(tensor.values, tensor.grads))
        return tensor
    
    @classmethod
    def ones(cls, shape, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = np.ones(shape, dtype = Flags.global_type())
        return tensor
    
    @classmethod
    def full(cls, shape, value, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = np.full(shape, value, dtype = Flags.global_type())
        return tensor
    
    @classmethod
    def randn(cls, shape, mean = 0, stddev = 1, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = np.random.normal(mean, stddev, shape) # will be casted by ensure_type
        return tensor
    
    @classmethod
    def random(cls, shape, min = 0, max = 1, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = np.random.rand(*shape) * (max - min) + min
        return tensor
    
    @classmethod
    def from_values(cls, values: _TensorArray, requires_grad: bool = False) -> Tensor:
        tensor = cls(values.shape, requires_grad)
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
    def values(self, other: _TensorArray) -> None:
        if not isinstance(other, np.ndarray):
            raise TypeError("other must be an array")
        self.__values = other

    @property
    def grads(self) -> _TensorArray:
        return self.__grads
    
    @grads.setter
    @ensure_type
    @ensure_shape
    def grads(self, other) -> None:
        if not isinstance(other, np.ndarray):
            raise TypeError("other must be an array")
        self.__grads = other
    
    @ensure_type
    def accumulate_grads(self, grad_updates: _TensorArray) -> None:
        if self.__requires_grad and not Flags.no_grad():
            self.grads += grad_updates

    def set_operation(self, operation: Operation) -> None:
        if isinstance(operation, type) and issubclass(operation, Operation):
            self.__op = operation
        else:
            raise TypeError(f"operation must be Operation type, not {type(operation)}")

    def add_children(self, *children) -> None:
        for child in children:
            self.__children.append(child)

    def forward(self) -> None:
        result_values = self.__op.forward(*(child.values for child in self.__children))
        self.values = result_values

    def backward(self) -> None:
        children_grads_update = self.__op.backward(
            self.grads,
            self.values,
            *(child.values for child in self.__children)
        )

        for child, grads_updates in zip(self.__children, children_grads_update):
            child.accumulate_grads(grads_updates) # update grads for every child

    # OPERATIONS
    def __mul__(self, other: Tensor) -> Tensor:
        result = Multiplication.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.__requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Multiplication)
        return tensor
    
    def __add__(self, other: Tensor) -> Tensor:
        result = Multiplication.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.__requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Addition)
        return tensor

