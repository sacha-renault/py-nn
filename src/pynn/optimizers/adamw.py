from collections.abc import Callable
from typing import Any

from .optimizer import Optimizer
from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp
from .adam import Adam


class AdamW(Adam):
    def __init__(self,
                 lr,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 fsaturate: Callable[[Any], Any] | None = None,
                 weight_decay=0.01) -> None:
        super().__init__(lr, beta1, beta2, epsilon, fsaturate)
        self._wd = weight_decay

    def _adam_step(self, param, m, v):
        # Perform the standard Adam step and get updated m and v
        m, v = super()._adam_step(param, m, v)

        # Apply weight decay directly to parameters
        param.values -= self._wd * param.values

        return m, v

