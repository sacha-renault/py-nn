from collections.abc import Callable
from .optimizer import Optimizer
from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, fsaturate: Callable[[xp.ndarray], xp.ndarray] | None = None) -> None:
        super().__init__(lr, fsaturate)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self.m_dw, self.v_dw = {}, {}
        self.m_db, self.v_db = {}, {}
        self.t = 1

    def _adam_step(self, param, m, v):
        m = self._beta1 * m + (1 - self._beta1) * param.grads
        v = self._beta2 * v + (1 - self._beta2) * xp.pow(param.grads, 2)
        m_corr = m / (1 - self._beta1 ** self.t)
        v_corr = v / (1 - self._beta2 ** self.t)
        param.values -= self.learning_rate * (m_corr / (xp.sqrt(v_corr) + self._epsilon))
        return m, v

    def _update_step(self, param: list[Tensor]) -> None:
        if isinstance(param, WeightTensor):
            m = self.m_dw.get(param, 0)
            v = self.v_dw.get(param, 0)
            self.m_dw[param], self.v_dw[param] = self._adam_step(param, m, v)
        elif isinstance(param, BiasTensor):
            m = self.m_db.get(param, 0)
            v = self.v_db.get(param, 0)
            self.m_db[param], self.v_db[param] = self._adam_step(param, m, v)
        else:
            raise TypeError(f"Adam cannot work with tensor type: {type(param)}, {param}")
