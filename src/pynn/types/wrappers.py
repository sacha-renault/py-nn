from ..flags import Flags

from functools import wraps
import numpy as np

def ensure_type(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # first convert tuple as list
        args = list(args)

        # 1st arg is self and should be a tensor
        dtype = Flags.global_type()

        # check that all other arrays are same dtype
        for i, arg in enumerate(args[1:], 1):
            if isinstance(arg, np.ndarray) and arg.dtype != dtype:
                args[i] = arg.astype(dtype, copy=False)

        # check also for kwargs 
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and value.dtype != dtype:
                kwargs[key] = value.astype(dtype, copy=False)

        return func(*args, **kwargs)
    return wrapper

def ensure_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # first convert tuple as list
        args = list(args)

        # 1st arg is self and should be a tensor
        tensor = args[0]
        shape = getattr(tensor, "shape")

        # check that all other arrays have the same shape
        for i, arg in enumerate(args[1:], 1):
            if isinstance(arg, np.ndarray) and arg.shape != shape:
                raise Exception(f"Shape mismatch for arg {i}. "
                                f"Tensor shape: {shape}, arg shape: {arg.shape}")

        # check also for kwargs 
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and value.shape != shape:
                raise Exception(f"Shape mismatch for kwarg '{key}'. "
                                f"Tensor shape: {shape}, kwarg shape: {value.shape}")

        return func(*args, **kwargs)
    return wrapper
