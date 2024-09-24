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


def im2col(tensor: Tensor, kernel_size: int, stride: int) -> Tensor:
    def forward(x):
        # Get input dimensions
        batch_size, input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width = kernel_size, kernel_size
        stride_height, stride_width = stride, stride

        # Calculate output dimensions
        output_height = (input_height - kernel_height) // stride_height + 1
        output_width = (input_width - kernel_width) // stride_width + 1

        # Calculate strides for the original tensor
        batch_stride, h_stride, w_stride, c_stride = x.strides
        # Strides for the new im2col view (no copying, just a view)
        new_shape = (batch_size, output_height, output_width, kernel_height, kernel_width, input_channels)
        new_strides = (batch_stride, h_stride * stride_height, w_stride * stride_width, h_stride, w_stride, c_stride)

        # Create strided view using as_strided (no data copying)
        col_view = xp.lib.stride_tricks.as_strided(x, shape=new_shape, strides=new_strides)

        # Ensure reshaping does not break the view and keep references
        col_flat = col_view.reshape(batch_size * output_height * output_width, kernel_height * kernel_width * input_channels)
        return col_flat

    def backward(parent_grad, parent_values, x):
        batch_size, input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width = kernel_size, kernel_size
        stride_height, stride_width = stride, stride

        # Calculate output dimensions
        output_height = (input_height - kernel_height) // stride_height + 1
        output_width = (input_width - kernel_width) // stride_width + 1

        # Initialize an empty tensor to store the gradient w.r.t. the input
        dx = xp.zeros_like(x)

        # Reshape the parent_grad to match the expected shape (batch_size, num_patches, kernel_size * input_channels)
        grad_reshaped = parent_grad.reshape(batch_size, output_height * output_width, kernel_height, kernel_width, input_channels)

        # Iterate over each patch and accumulate gradients back to the input tensor
        col_idx = 0
        for i in range(output_height):
            for j in range(output_width):
                # Extract the grad patch correctly
                grad_patch = grad_reshaped[:, col_idx, :, :, :]  # Patch already reshaped to (batch_size, kernel_height, kernel_width, input_channels)
                
                # Accumulate gradients back to the original tensor
                dx[:, 
                i * stride_height : i * stride_height + kernel_height, 
                j * stride_width : j * stride_width + kernel_width, 
                :] += grad_patch
                col_idx += 1

        return dx,

    # Define the operation using LambdaOperation
    op = LambdaOperation(forward, backward)

    # Perform the forward pass
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)

    # Add the original tensor as a child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)

    return result_tensor

