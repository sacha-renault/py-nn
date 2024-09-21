from src.pynn.tensor.tensor import Tensor
import numpy as np

t1 = Tensor.full((10,15), 5)
t2 = Tensor.from_values(np.arange(150).reshape(10, 15))
tensor = t1 * t2
print(np.min(tensor.values), np.max(tensor.values))
print(tensor.values)

# tensor.values = np.array(np.random.rand(11, 10))