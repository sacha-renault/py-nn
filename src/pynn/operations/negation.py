from .. import xp
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Negation(Operation):
    @staticmethod
    def forward(x):
        return -x

    @staticmethod
    def backward(grad_output, parent_values, x):
        return -grad_output