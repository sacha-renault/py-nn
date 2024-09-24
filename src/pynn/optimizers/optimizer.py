from abc import ABC, abstractmethod
from collections.abc import Callable

from ..flags import Flags
from ..tensor import Tensor
from ..tensor.tensor_view import _TensorView
from .. import xp

class Optimizer(ABC):
    def __init__(self, lr, fsaturate: Callable[[xp.ndarray], xp.ndarray] | None = None) -> None:
        self.learning_rate = lr
        self._fsaturate = fsaturate

    @property
    def learning_rate(self) -> float:
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        self._lr = Flags.global_type()(value) # cast into global type

    def _fsaturate_wrapper(self, param: Tensor) -> None:
        if self._fsaturate is not None:
            xp.copyto(param.grads, self._fsaturate(param.grads))

    def update(self, params: list[Tensor]) -> None:
        for param in params:
            self._fsaturate_wrapper(param) # saturate (or basic clip)
            # TODO regularization technique ?
            self._update_step(param)
        self._loop_done()

    def _loop_done(self) -> None: pass

    @abstractmethod
    def _update_step(self, param: Tensor) -> None: ...
