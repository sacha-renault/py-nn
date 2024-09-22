from typing import Any
from abc import ABC, abstractmethod
from .. import xp
from ..tensor import Tensor

class Layer(ABC):
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor: ...

    @property
    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @parameters.setter
    @abstractmethod
    def parameters(self, params: list[Tensor]) -> None: ...