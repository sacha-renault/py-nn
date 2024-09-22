from ..tensor import Tensor
from ..math import log, reduce_mean, reduce_sum, clip, pow, abs

def categorical_cross_entory(pred: Tensor, real: Tensor, epsilon: float = 1e-9) -> Tensor:
    # prevent log(0)
    pred = clip(pred, epsilon, 1.0 - epsilon)

    # y_real * log(pred)
    log_pred = real * log(pred)

    # negative sum over input axis
    cce = - reduce_sum(log_pred, axis = 1)

    # return the mean of the output
    return reduce_mean(cce)

def mean_square_error(pred: Tensor, real: Tensor) -> Tensor:
    return pow(pred - real, 2)

def mean_absolute_error(pred: Tensor, real: Tensor) -> Tensor:
    return abs(pred - real)

