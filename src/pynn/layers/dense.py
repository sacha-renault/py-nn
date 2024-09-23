from typing import Any
from collections.abc import Callable
from .layer import Layer
from .. import xp
from ..tensor import Tensor
from ..tensor.sub_class import BiasTensor, WeightTensor
from ..math import dot
from ..types import auto_convert_to_cupy
from ..utils.initializer import initializer_xavier_relu

class Dense(Layer):
    def __init__(self, 
                input_features: int, 
                output_features: int,
                activation = None,
                stddev_initializer: xp.generic | Callable[[int, int], float] | None = None) -> None:
        # get stddev value
        if stddev_initializer is None:
            stddev = initializer_xavier_relu(input_features, output_features)
        elif callable(stddev_initializer):
            stddev = stddev_initializer(input_features, output_features)
        else:
            stddev = stddev_initializer

        # init weights
        self._wi = WeightTensor.randn(
            (input_features, output_features), 
            stddev, 
            requires_grad=True)
        
        # init biases
        self._bias = BiasTensor.zeros((output_features, ), requires_grad=True)

        # activation
        self._activation = activation

    def __call__(self, x: Tensor) -> Tensor | Any:
        output = dot(x, self._wi) + self._bias
        if self._activation:
            return self._activation(output)
        return output
    
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
        
