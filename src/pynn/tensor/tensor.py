import numpy as np
from ..types import (
    _float16, _float32, _float64,
    _tensorArray16, _tensorArray32, _tensorArray64,
    _tensorArray,
    ensure_type, ensure_shape
)

class Tensor:
    def __init__(self, shape, dtype = _float32) -> None:
        self.__dtype = dtype
        self.__values = np.empty(shape, dtype = dtype)
        self.__grads = np.empty(shape, dtype = dtype)

    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def shape(self):
        return self.__values.shape # we could use grad but it's the same in the end

    @property
    def values(self) -> _tensorArray:
        return self.__values
    
    @values.setter
    @ensure_type
    @ensure_shape
    def values(self, other) -> None:
        self.__values = other
    
