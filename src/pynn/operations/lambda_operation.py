import numpy as np
from collections.abc import Callable
from typing import Any
from .base_operation import Operation
from ..utils import collapse_broadcast
from ..types import _TensorArray

class LambdaOperation(Operation):
    def __init__(self, forward: Callable[[Any], _TensorArray], backward: Callable[[Any], _TensorArray]):
        self.__backward = backward
        self.__forward = forward

    def forward(self, *children_values) -> _TensorArray:
        return self.__forward(*children_values)
    
    @collapse_broadcast
    def backward(self, parent_grad, parent_values, *children_values) -> tuple[_TensorArray]:
        return self.__backward(parent_grad, parent_values, *children_values)