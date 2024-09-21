from .. import xp
from ..operations import LambdaOperation, Addition
from ..tensor import Tensor

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