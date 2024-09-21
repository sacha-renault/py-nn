from src.pynn.tensor.tensor import Tensor
import numpy as np
from src.pynn.flags import Flags
from src.pynn.graph import ComputeGraph
from src.pynn.math import tanh


t1 = Tensor.full((5,5), 1, requires_grad=True)
t2 = Tensor.from_values(np.array([5]), requires_grad=True)
tensor = t1 * t2
tensor2 = tensor - Tensor.full((5,5), 3, requires_grad=True)
output = tensor2
output = tanh(tensor2)


graph = ComputeGraph(output)
output.grads = np.ones((5,5))
graph.backward()


print(output.values)
print(t1.grads)





# tensor.values = np.array(np.random.rand(11, 10))