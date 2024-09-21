import numpy as np
from ..operations import LambdaOperation
from ..tensor import Tensor

def tanh(tensor: Tensor) -> Tensor:
    def forward(x):
        return np.tanh(x)  # Forward pass for tanh

    def backward(parent_grad, parent_values, x):
        # Use parent_values which already contains tanh(x)
        grad_x = parent_grad * (1 - parent_values ** 2)  # Derivative of tanh
        return grad_x,  # Return a tuple with the gradient

    # Create the LambdaOperation for tanh
    op = LambdaOperation(forward, backward)
    
    # Perform the forward pass
    result = op.forward(tensor.values)
    
    # Create a new tensor with the result
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    
    # Add the current tensor as the child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor