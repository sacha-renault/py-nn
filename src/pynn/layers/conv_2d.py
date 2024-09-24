from typing import Any
from collections.abc import Callable
from .layer import Layer
from .. import xp
from ..tensor import Tensor
from ..tensor.sub_class import BiasTensor, WeightTensor
from ..math import dot, im2col
from ..types import auto_convert_to_cupy, _TensorArray
from ..utils.initializer import he_initializer


class Conv2d(Layer):
    def __init__(self,
                 num_channel_int: int,
                 num_channel_out: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 activation = None) -> None:
        self._num_channel_int = num_channel_int
        self._num_channel_out = num_channel_out
        self._kernel_size = kernel_size
        self._strides = strides
        self._activation = activation

        # init filters
        num = self._kernel_size * self._kernel_size * self._num_channel_int
        limit = (6 / (self._num_channel_int + self._num_channel_out)) ** 0.5
        self._filters = WeightTensor.random((num, self._num_channel_out), -limit, limit, requires_grad=True)



    def __call__(self, tensor: Tensor) -> Tensor:
        # calculate the shape of the output tensor
        input_shape = tensor.shape
        batch_size, input_height, input_width, input_channels = input_shape

        # Calculate output height and width
        output_height = (input_height - self._kernel_size) // self._strides + 1
        output_width = (input_width - self._kernel_size) // self._strides + 1

        # process convolution
        cols = im2col(tensor, self._kernel_size, self._strides)
        cols.retains_grad()
        conv_output = dot(cols, self._filters)

        # if activation
        if self._activation:
            conv_output = self._activation(conv_output)

        # Reshape conv_output to the correct output shape
        new_shape = (batch_size, output_height, output_width, self._num_channel_out)
        reshaped = conv_output.reshape(new_shape)
        return reshaped


    @property
    def parameters(self) -> list[Tensor]:
        return [self._filters]

    @parameters.setter
    def parameters(self, params: list[Tensor]) -> None:
        # params[0] shoudl repr the filters
        # no need to ensure shape, setter of values already does
        if isinstance(params[0], xp.ndarray):
            self._filters.values = params[0]
        elif isinstance(params[0], Tensor):
            self._filters.values = params[0].values
        else:
            raise TypeError("to set param, should use Tensor or ndarray")


