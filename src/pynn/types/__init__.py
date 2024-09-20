from numpy import typing as npt
import numpy as np

from .wrappers import ensure_type, ensure_shape

# float type
_float64 = np.float64
_float32 = np.float32
_float16 = np.float16

# array types
_tensorArray64 = npt.NDArray[_float64]
_tensorArray32 = npt.NDArray[_float32]
_tensorArray16 = npt.NDArray[_float16]
_tensorArray = npt.NDArray