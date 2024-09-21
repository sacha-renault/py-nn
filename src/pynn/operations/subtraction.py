from .. import xp
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Subtraction(Operation):
    @staticmethod
    def forward(*children_values):
        if len(children_values) != 2:
            raise ValueError("Subtraction requires exactly 2 Tensors")
        return xp.subtract(*children_values)
    
    @staticmethod
    @collapse_broadcast
    def backward(parent_grad: _TensorArray, 
                 parent_values: _TensorArray, 
                 *children_values: _TensorArray):
        if len(children_values) != 2:
            raise ValueError("Subtraction requires exactly 2 Tensors")
        
        # Gradient w.r.t. first tensor is positive; second tensor is negative
        child_grad0 = parent_grad
        child_grad1 = -parent_grad
        return (child_grad0, child_grad1)
