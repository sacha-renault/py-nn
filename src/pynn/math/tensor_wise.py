from .. import xp
from ..operations import LambdaOperation, Addition
from ..tensor import Tensor
from ..utils import collapse_broadcast, get_expanded_reduced_shape
from ..utils.broadcast import _collapse_broadcast

def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    def forward(x, y):
        return xp.dot(x, y)  # Forward pass using NumPy's dot function

    def backward(parent_grad, parent_values, x, y):
        # Derivatives of dot product:
        # Gradient w.r.t. the first tensor is parent_grad * y.T
        # Gradient w.r.t. the second tensor is parent_grad.T * x
        grad_x = xp.dot(parent_grad, y.T)
        grad_y = xp.dot(x.T, parent_grad)
        return grad_x, grad_y

    # Create the LambdaOperation for dot
    op = LambdaOperation(forward, backward)
    result = op.forward(tensor1.values, tensor2.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    # Add the current tensors as children and set the operation
    result_tensor.add_children(tensor1, tensor2)
    result_tensor.set_operation(op)
    return result_tensor

def reduce_sum(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    def forward(x):
        return xp.sum(x, axis=axis, keepdims=keepdims)  # Forward pass using NumPy or CuPy sum

    def backward(parent_grad, parent_values, x):
        # calculate target shape
        target_shape = get_expanded_reduced_shape(x.shape, axis)

        # apply
        reshaped = xp.reshape(parent_grad, target_shape)
        return xp.broadcast_to(reshaped, x.shape),

    # Create the LambdaOperation for reduce_sum
    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    
    # Add the original tensor as a child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    return result_tensor

def reduce_mean(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    def forward(x):
        return xp.mean(x, axis=axis, keepdims=keepdims)  # Forward pass using NumPy or CuPy mean

    def backward(parent_grad, parent_values, x):
        # calculate target shape
        target_shape = get_expanded_reduced_shape(x.shape, axis)

        # apply
        reshaped = xp.reshape(parent_grad, target_shape)

        # Broadcast the gradient to the original shape and divide by the number of elements reduced
        if axis is None:
            grad_x = xp.broadcast_to(reshaped, x.shape) / x.size  # Total size of the array
        else:
            grad_x = xp.broadcast_to(reshaped, x.shape) / xp.prod(x.shape[axis])  # Size along specific axis
        return grad_x,

    # Create the LambdaOperation for reduce_mean
    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    
    # Add the original tensor as a child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    return result_tensor

def softmax(tensor: Tensor, axis: int = -1) -> Tensor:
    def forward(x):
        x_max = xp.max(x, axis=-1, keepdims=True)
        exp_x = xp.exp(x - x_max)
        return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)

    def backward(parent_grad, parent_values, x):
        s = parent_values  # This is the softmax result from the forward step
        sum_term = xp.sum(parent_grad * s, axis=axis, keepdims=True)
        grad_x = s * (parent_grad - sum_term)
        return grad_x,


    # Create the LambdaOperation for softmax
    # Ensure tensor is shape (bs, input)
    # softmax should apply on axis -1
    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    
    # Add the original tensor as a child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    return result_tensor
