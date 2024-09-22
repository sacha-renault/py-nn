from typing import Any
from .layer import Layer
from .. import xp
from ..tensor import Tensor
from ..math import dot
from ..types import auto_convert_to_cupy

class Dense(Layer):
    def __init__(self, 
                input_features: int, 
                output_features: int) -> None:
        # init weights
        self._wi = Tensor.randn(
            (input_features, output_features), 
            (2/(input_features * output_features))**(1/2), 
            requires_grad=True)
        
        # init biases
        self._bias = Tensor.zeros((output_features, ), requires_grad=True)

    def __call__(self, x: Tensor) -> Any:
        return dot(x, self._wi) + self._bias
    
    @property
    def parameters(self) -> list[Tensor]:
        return [self._wi, self._bias]

    @parameters.setter
    @auto_convert_to_cupy
    def parameters(self, params: list[Tensor] | list[xp.ndarray]) -> None:
        # params[0] shoudl repr the wi
        # no need to ensure shape, setter of values already does
        if isinstance(params[0], xp.ndarray):
            self._wi.values = params[0]
        elif isinstance(params[0], Tensor):
            self._wi.values = params[0].values
        else:
            raise TypeError("to set param, should use Tensor or ndarray")
        

        # params[1] shoudl repr the bias
        # no need to ensure shape, setter of values already does
        if isinstance(params[1], xp.ndarray):
            self._bias.values = params[1]
        elif isinstance(params[1], Tensor):
            self._bias.values = params[1].values
        else:
            raise TypeError("to set param, should use Tensor or ndarray")
        
