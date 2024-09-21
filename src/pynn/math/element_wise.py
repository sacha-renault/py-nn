from .. import xp
from ..operations import LambdaOperation
from ..tensor import Tensor

def tanh(tensor: Tensor) -> Tensor:
    def forward(x):
        return xp.tanh(x)  # Forward pass for tanh

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

def pow(tensor: Tensor, exponent: float) -> Tensor:
    def forward(x):
        return xp.power(x, exponent)  # Forward pass for power

    def backward(parent_grad, parent_values, x):
        # Derivative: exponent * x^(exponent - 1)
        grad_x = parent_grad * exponent * xp.power(x, exponent - 1)
        return grad_x,  # Return a tuple with the gradient

    # Create the LambdaOperation for pow
    op = LambdaOperation(forward, backward)
    
    # Perform the forward pass
    result = op.forward(tensor.values)
    
    # Create a new tensor with the result
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    
    # Add the current tensor as the child and set the operation
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor

def square(tensor: Tensor) -> Tensor:
    return pow(tensor, 2)

def sqrt(tensor: Tensor) -> Tensor:
    return pow(tensor, 1 / 2)

def abs(tensor: Tensor) -> Tensor:
    def forward(x):
        return xp.abs(x)  # Forward pass for abs

    def backward(parent_grad, parent_values, x):
        # Derivative of abs: 1 where x > 0, -1 where x < 0
        grad_x = parent_grad * xp.sign(x)
        return grad_x,

    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor

def exp(tensor: Tensor) -> Tensor:
    def forward(x):
        return xp.exp(x)  # Forward pass for exp

    def backward(parent_grad, parent_values, x):
        # Derivative of exp is exp(x)
        grad_x = parent_grad * xp.exp(x)
        return grad_x,

    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor

def log(tensor: Tensor) -> Tensor:
    # TODO check why it returns NAN
    # (probably if there is any value < 0)

    def forward(x):
        return xp.log(x)  # Forward pass for log

    def backward(parent_grad, parent_values, x):
        # Derivative of log(x) is 1/x
        grad_x = parent_grad / x
        return grad_x,

    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor

def relu(tensor: Tensor) -> Tensor:
    def forward(x):
        return xp.maximum(0, x)  # Forward pass for ReLU

    def backward(parent_grad, parent_values, x):
        # Derivative of ReLU: 1 where x > 0, 0 where x <= 0
        grad_x = parent_grad * (x > 0).astype(x.dtype)
        return grad_x,

    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor

def clip(tensor: Tensor, min_value: float = 0, max_value: float = 1) -> Tensor:
    def forward(x):
        return xp.clip(x, min_value, max_value)  # Forward pass for clip

    def backward(parent_grad, parent_values, x):
        # Derivative of clip: 1 where min_value < x < max_value, 0 otherwise
        grad_x = parent_grad * ((x >= min_value) & (x <= max_value)).astype(x.dtype)
        return grad_x,

    op = LambdaOperation(forward, backward)
    result = op.forward(tensor.values)
    result_tensor = Tensor.from_values(result, requires_grad=tensor.requires_grad)
    result_tensor.add_children(tensor)
    result_tensor.set_operation(op)
    
    return result_tensor