from functools import wraps
from .. import xp
from ..types import _TensorArray

def _find_broadcast_axes(shape_a, shape_b):
    if isinstance(shape_a, int):
        return list(range(len(shape_b)))
    
    if isinstance(shape_b, int):
        return list(range(len(shape_a)))

    l = len(shape_a) - 1
    r = len(shape_b) - 1
    axes = []

    while l >= 0 or r >= 0:
        dim_a = shape_a[l] if l >= 0 else 1
        dim_b = shape_b[r] if r >= 0 else 1

        if dim_a == dim_b:
            pass  # No broadcasting needed on this axis
        elif dim_a == 1 or dim_b == 1:
            # Broadcasting occurs on this axis
            axes.append(max(l, r))
        else:
            # Dimensions are incompatible
            raise ValueError(f"Incompatible dimensions at positions {l} and {r}: {dim_a} vs {dim_b}")

        l -= 1
        r -= 1

    return axes[::-1]  # Return axes in increasing order

def _collapse_broadcast(broadcasted_array: _TensorArray, original_shape) -> _TensorArray:
    if broadcasted_array.shape == original_shape:
        return broadcasted_array # there was no change
    expanded_axes = _find_broadcast_axes(broadcasted_array.shape, original_shape)
    return xp.sum(broadcasted_array, axis = tuple(expanded_axes))

def collapse_broadcast(func):
    @wraps(func)
    def wrapper(*args):
        # perform function normally
        result = func(*args)
        
        # Handle the case where the result is a tuple or list of gradients
        if isinstance(result, (tuple, list)):
            return [ _collapse_broadcast(res, arg.shape) for res, arg in zip(result, args[-len(result):]) ]
        else:
            # For single tensor return values (just in case)
            return _collapse_broadcast(result, args[-1].shape)
    return wrapper

        