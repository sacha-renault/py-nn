import numpy as np
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Multiplication(Operation):
    @staticmethod
    def forward(*children_values):
        if len(children_values) != 2:
            raise ValueError("Multiplication can only occure with 2 Tensor")
        return np.multiply(*children_values)
    
    @staticmethod
    @collapse_broadcast
    def backward(parent_grad: _TensorArray, 
                 parent_values: _TensorArray, 
                 *children_values: _TensorArray):
        if len(children_values) != 2:
            raise ValueError("Multiplication can only occure with 2 Tensor")
        child_grad0 = parent_grad * children_values[1]
        child_grad1 = parent_grad * children_values[0]
        return (child_grad0, child_grad1)