from collections.abc import Callable

from .optimizer import Optimizer
from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp


class RMSProp(Optimizer):
    def __init__(self, lr, alpha=0.9, epsilon=1e-8, fsaturate: Callable[[xp.ndarray], xp.ndarray] | None = None) -> None:
        super().__init__(lr, fsaturate)
        self._alpha = alpha
        self._epsilon = epsilon
        self._s_dw = {}

    def _update_step(self, param: Tensor) -> None:
        m = self._s_dw.get(param, 0)
        m = self._alpha * m + (1 - self._alpha) * xp.square(param.grads)
        xp.copyto(param.values, param.values - self.learning_rate * param.grads / (xp.sqrt(m) + self._epsilon))
        self._s_dw[param] = m
