import numbers

from . import Tensor
from .. import xp
from ..types import (
    _float16, _float32, _float64,
    _TensorArray,
    ensure_type, ensure_shape, auto_convert_to_cupy
)

class _TensorView(Tensor):
    def __init__(self, ref_tensor: Tensor, slices: list[slice]) -> None:
        self.__ref_tensor = ref_tensor
        self.__slices = slices
        self._op = None

    @property
    def shape(self):
        return self.__ref_tensor.values[*self.__slices].shape
    
    @property
    def children(self):
        return self.__ref_tensor.children
    
    @property
    def requires_grad(self) -> bool:
        return self.__ref_tensor.requires_grad

    @property
    def values(self) -> _TensorArray:
        return self.__ref_tensor.values[*self.__slices]
    
    @values.setter
    @ensure_type
    @ensure_shape
    def values(self, other: _TensorArray) -> None:
        if isinstance(other, (numbers.Number, xp.generic)):
            other = xp.array([other]) # convert into (1,)
        elif not isinstance(other, xp.ndarray):
            raise TypeError("other must be an array")
        self.__ref_tensor.values[*self.__slices] = other

    @property
    def grads(self) -> _TensorArray:
        return self.__ref_tensor.grads[*self.__slices]
    
    @grads.setter
    @ensure_type
    @ensure_shape
    def grads(self, other: _TensorArray) -> None:
        if isinstance(other, (numbers.Number, xp.generic)):
            other = xp.array([other]) # convert into (1,)
        elif not isinstance(other, xp.ndarray):
            raise TypeError("other must be an array")
        self.__ref_tensor.grads[*self.__slices] = other
    
