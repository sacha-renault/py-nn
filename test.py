from src.pynn.tensor.tensor import Tensor
import numpy as np

t1 = Tensor.full((5,5), 5, requires_grad=True)
t2 = Tensor.from_values(np.array([0]), requires_grad=True)
tensor = t1 * t2
print(np.min(tensor.values), np.max(tensor.values))
tensor.grads = np.ones((5,5))
print(tensor.grads)

tensor.backward()
print(t1.grads)
print(t2.grads)

# tensor.values = np.array(np.random.rand(11, 10))