from collections.abc import Callable
from typing import Any

from .optimizer import Optimizer
from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp
from .adam import Adam


class NAdam(Adam):
    def _adam_step(self, param, m, v):
        # Update m and v using the standard Adam update
        m = self._beta1 * m + (1 - self._beta1) * param.grads
        v = self._beta2 * v + (1 - self._beta2) * xp.pow(param.grads, 2)

        # Bias-corrected estimates
        m_corr = m / (1 - self._beta1 ** self.t)
        v_corr = v / (1 - self._beta2 ** self.t)

        # Nesterov lookahead
        lookahead_m = self._beta1 * m_corr + (1 - self._beta1) * param.grads

        # Update parameters using the Nesterov-corrected momentum
        param.values -= self.learning_rate * lookahead_m / (xp.sqrt(v_corr) + self._epsilon)

        # Return the updated m and v for future use
        return m, v

