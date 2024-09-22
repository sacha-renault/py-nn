from ..tensor import Tensor
from ..math import log, reduce_mean, reduce_sum, clip

def categorical_cross_entory(pred: Tensor, real: Tensor, epsilon: float = 1e-9) -> Tensor:
    # prevent log(0)
    pred = clip(pred, epsilon, 1.0 - epsilon)

    # y_real * log(pred)
    log_pred = real * log(pred)

    # sum over input axis
    cce = reduce_sum(log_pred, axis = 1)

    # return the negative mean of the output
    return reduce_mean(cce)

