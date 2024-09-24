from ..tensor import Tensor
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr) -> None:
        self._lr = lr

    def update(self, params: list[Tensor]) -> None:
        for param in params:
            param.values = param.values - self.learning_rate * param.grads