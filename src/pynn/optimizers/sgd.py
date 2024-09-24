from ..tensor import Tensor
from .optimizer import Optimizer
from .. import xp

class SGD(Optimizer):
    def _update_step(self, param: Tensor) -> None:
        xp.copyto(param.values, param.values - self.learning_rate * param.grads)