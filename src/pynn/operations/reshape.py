from .. import xp
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class Reshape(Operation):
    def __init__(self, base_shape, next_shape) -> None:
        self._base_shape = base_shape
        self._next_shape = next_shape

    def forward(self, *children_values):
        if len(children_values) != 1:
            raise ValueError("Forward requires exactly one child")
        return children_values[0].reshape(self._next_shape)
    
    def backward(self,
                 parent_grad: _TensorArray, 
                 parent_values: _TensorArray, 
                 *children_values: _TensorArray):
        if len(children_values) != 1:
            raise ValueError("Forward requires exactly one child")
        
        return parent_grad.reshape(self._base_shape)
