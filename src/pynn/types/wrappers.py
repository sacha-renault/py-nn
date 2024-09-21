from functools import wraps

import numpy as np

from .. import xp
from ..flags import Flags

def ensure_type(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # first convert tuple as list
        args = list(args)

        # 1st arg is self and should be a tensor
        dtype = Flags.global_type()

        # check that all other arrays are same dtype
        for i, arg in enumerate(args[1:], 1):
            if isinstance(arg, xp.ndarray) and arg.dtype != dtype:
                args[i] = arg.astype(dtype, copy=False)

        # check also for kwargs 
        for key, value in kwargs.items():
            if isinstance(value, xp.ndarray) and value.dtype != dtype:
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
            if isinstance(arg, xp.ndarray) and arg.shape != shape:
                raise Exception(f"Shape mismatch for arg {i}. "
                                f"Tensor shape: {shape}, arg shape: {arg.shape}")

        # check also for kwargs 
        for key, value in kwargs.items():
            if isinstance(value, xp.ndarray) and value.shape != shape:
                raise Exception(f"Shape mismatch for kwarg '{key}'. "
                                f"Tensor shape: {shape}, kwarg shape: {value.shape}")

        return func(*args, **kwargs)
    return wrapper

def auto_convert_to_cupy(func):
    if not Flags.using_cuda():  # Assuming Flags.using_cuda() determines if CUDA is used
        print("normal")
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert tuple as list to allow modification
        args = list(args)

        # Check that all positional arguments are numpy arrays, convert to cupy
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = xp.asarray(arg)  # Convert numpy to cupy

        # Check for keyword arguments that are numpy arrays and convert them to cupy
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                kwargs[key] = xp.asarray(value)  # Convert numpy to cupy

        return func(*args, **kwargs)
    return wrapper
