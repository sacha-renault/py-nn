import numpy as np
from .base_operation import Operation

class Mul(Operation):
    @staticmethod
    def forward(*children_values):
        if len(children_values) != 2:
            raise ValueError("Multiplication can only occure with 2 Tensor")
        return np.multiply(*children_values)
    
    @staticmethod
    def backward(parent_grad, parent_values, *children_values):
        return None