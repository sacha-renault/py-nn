from numpy import typing as npt
import numpy as np

from .wrappers import ensure_type, ensure_shape

# float type
_float64 = np.float64
_float32 = np.float32
_float16 = np.float16

# array types
_TensorArray64 = npt.NDArray[_float64]
_TensorArray32 = npt.NDArray[_float32]
_TensorArray16 = npt.NDArray[_float16]
_TensorArray = npt.NDArray