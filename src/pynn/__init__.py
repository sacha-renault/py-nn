import os
import warnings

# Default to NumPy
xp = None  # `xp` will be either numpy or cupy, depending on what is available or set

# Allow user to control via environment variable (optional)
USE_CUDA = os.getenv("USE_CUDA", "False").lower() in ("true", "1")

if USE_CUDA:
    try:
        import cupy as xp
    except ImportError:
        warnings.warn("cupy couldn't be loaded, fallback on numpy")
        import numpy as xp
else:
    import numpy as xp  # Fallback to NumPy

# You can now use xp for array operations and it will be either NumPy or CuPy
