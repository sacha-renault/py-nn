from typing import TypeAlias
from .. import xp

from .wrappers import ensure_type, ensure_shape, auto_convert_to_cupy

# float type
_float64: TypeAlias = xp.float64
_float32: TypeAlias = xp.float32
_float16: TypeAlias = xp.float16

# array types
# _TensorArray64 = xp.ndarray[_float64]
# _TensorArray32 = xp.ndarray[_float32]
# _TensorArray16 = xp.ndarray[_float16]
_TensorArray: TypeAlias = xp.ndarray

# At initialization, type will be float32
from ..flags import Flags

Flags.set_global_type(_float32)