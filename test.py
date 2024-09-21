from src.pynn.tensor.tensor import Tensor
import numpy as np
from src.pynn.flags import Flags
from src.pynn.graph import ComputeGraph


t1 = Tensor.full((5,5), 1, requires_grad=True)
t2 = Tensor.from_values(np.array([5]), requires_grad=True)
tensor = t1 * t2
tensor2 = tensor + Tensor.full((5,5), 1, requires_grad=True)
graph = ComputeGraph(tensor2)


tensor2.grads = np.ones((5,5))

graph.backward()

print(t1.grads)
print(t2.grads)
print(tensor.grads)

t1.values = np.random.rand(5,5)
graph.forward()





# tensor.values = np.array(np.random.rand(11, 10))