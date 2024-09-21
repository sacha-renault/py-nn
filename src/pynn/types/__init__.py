from .. import xp

from .wrappers import ensure_type, ensure_shape

# float type
_float64 = xp.float64
_float32 = xp.float32
_float16 = xp.float16

# array types
_TensorArray64 = xp.ndarray[_float64]
_TensorArray32 = xp.ndarray[_float32]
_TensorArray16 = xp.ndarray[_float16]
_TensorArray = xp.ndarray

# At initialization, type will be float32
from ..flags import Flags
Flags.set_global_type(_float32)