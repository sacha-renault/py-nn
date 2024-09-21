from .. import xp
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Division(Operation):
    @staticmethod
    def forward(*children_values):
        if len(children_values) != 2:
            raise ValueError("Division requires exactly 2 Tensors")
        return xp.divide(*children_values)  # Element-wise division
    
    @staticmethod
    @collapse_broadcast
    def backward(parent_grad: _TensorArray, 
                 parent_values: _TensorArray, 
                 *children_values: _TensorArray):
        if len(children_values) != 2:
            raise ValueError("Division requires exactly 2 Tensors")
        
        # Gradient w.r.t. the first tensor (x/y) is 1/y, and w.r.t. the second tensor is -x/y^2
        x, y = children_values
        child_grad0 = parent_grad / y
        child_grad1 = -parent_grad * x / (y ** 2)
        return (child_grad0, child_grad1)
