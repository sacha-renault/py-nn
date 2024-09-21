def _find_broadcast_axes(shape_a, shape_b):
    if isinstance(shape_a, int):
        return list(range(len(shape_b)))
    
    if isinstance(shape_b, int):
        return list(range(len(shape_a)))

    l = len(shape_a) - 1
    r = len(shape_b) - 1
    axes = []

    while l >= 0 or r >= 0:
        print(f"l : {l}, r {r}")
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
        