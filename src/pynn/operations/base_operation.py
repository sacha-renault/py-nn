from abc import ABC, abstractmethod

class Operation(ABC):
    @staticmethod
    @abstractmethod
    def forward(*children_values): ...

    @staticmethod
    @abstractmethod
    def backward(parent_grad, parent_values, *children_values): ...