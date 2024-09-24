from abc import ABC, abstractmethod
from ..flags import Flags
from ..tensor import Tensor

class Optimizer(ABC):
    def __init__(self, lr) -> None:
        self.learning_rate = lr

    @property
    def learning_rate(self) -> float:
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value) -> float:
        self._lr = Flags.global_type()(value) # cast into global type

    @abstractmethod
    def update(self, params: list[Tensor]) -> None: ...
