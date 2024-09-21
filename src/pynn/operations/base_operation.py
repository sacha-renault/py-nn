from abc import ABC, abstractmethod

from ..types import _TensorArray

class Operation(ABC):
    @staticmethod
    @abstractmethod
    def forward(*children_values) -> _TensorArray: ...

    @staticmethod
    @abstractmethod
    def backward(parent_grad, parent_values, *children_values) -> tuple[_TensorArray]: ...