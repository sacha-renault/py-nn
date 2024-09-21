import numpy as np
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Addition(Operation):
    @staticmethod
    def forward(*children_values):
        if len(children_values) < 2:
            raise ValueError("Addition requires at least 2 Tensors")
        return np.add.reduce(children_values)  # Sum all input tensors
    
    @staticmethod
    @collapse_broadcast
    def backward(parent_grad: _TensorArray, 
                 parent_values: _TensorArray, 
                 *children_values: _TensorArray):
        if len(children_values) < 2:
            raise ValueError("Addition requires at least 2 Tensors")
        
        # The gradient is passed equally to all children, respecting broadcasting
        child_grads = [
            parent_grad for _ in children_values
        ]
        return tuple(child_grads)
