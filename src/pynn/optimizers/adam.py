from .optimizer import Optimizer
from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__(lr)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self.m_dw, self.v_dw = [], []
        self.m_db, self.v_db = [], []
        self.t = 1

    def _adam_step(self, param, m, v):
        m = self._beta1 * m + (1 - self._beta1) * param.grads
        v = self._beta2 * v + (1 - self._beta2) * xp.pow(param.grads, 2)
        m_corr = m / (1 - self._beta1 ** self.t)
        v_corr = v / (1 - self._beta2 ** self.t)
        param.values -= self.learning_rate * (m_corr / (xp.sqrt(v_corr) + self._epsilon))
        return m, v

    def update(self, params: list[Tensor]) -> None:
        if self.t == 1:
            for param in params:
                if isinstance(param, WeightTensor):
                    self.m_dw.append(0)
                    self.v_dw.append(0)
                elif isinstance(param, BiasTensor):
                    self.m_db.append(0)
                    self.v_db.append(0)
                else:
                    raise TypeError(f"Adam cannot work with tensor type: {type(param)}")

        i = 0
        j = 0
        for param in params:
            if isinstance(param, WeightTensor):
                self.m_dw[i], self.v_dw[i] = self._adam_step(param, self.m_dw[i], self.v_dw[i])
                i += 1
            elif isinstance(param, BiasTensor):
                self.m_db[j], self.v_db[j] = self._adam_step(param, self.m_db[j], self.v_db[j])
                j += 1
            else:
                raise TypeError(f"Adam cannot work with tensor type: {type(param)}")
