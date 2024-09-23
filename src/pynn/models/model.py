from typing import Any
from ..tensor import Tensor
from ..graph import ComputeGraph

class Model:
    def __init__(self, inputs: Tensor | list[Tensor], outputs: Tensor | list[Tensor]) -> None:
        # Ensure type for inputs
        if isinstance(inputs, Tensor):
            self.__inputs = [inputs]
        elif isinstance(inputs, list) and all(isinstance(x, Tensor) for x in inputs):
            self.__inputs = inputs
        else:
            raise TypeError("Inputs should be a Tensor or a list of Tensors")

        # Ensure type for outputs
        if isinstance(outputs, Tensor):
            self.__outputs = [outputs]
        elif isinstance(outputs, list) and all(isinstance(x, Tensor) for x in outputs):
            self.__outputs = outputs
        else:
            raise TypeError("Outputs should be a Tensor or a list of Tensors")
        
        # build a compute graph
        self._graph = ComputeGraph(self.__outputs)

        # ensure all inputs are included in the graph
        if not all([self._graph.is_tensor_in(x) for x in self.__inputs]):
            raise Exception("Broken graph") # TODO replace with an other exception
        
    @property
    def parameters(self) -> list[Tensor]:
        return self._graph.parameters
    
    def auto_grad(self) -> None:
        self._graph.backward()

    def zero_grad(self) -> None:
        self._graph.zero_grad()
        
    def __call__(self, *args: Tensor) -> Tensor | list[Tensor]:
        if len(args) != len(self.__inputs):
            raise Exception(f"Model has {len(self.__inputs)} inputs, but {len(args)} were provided.")
        
        # change input of the graph
        for model_in, arg in zip(self.__inputs, args):
            model_in.values = arg.values

        # execute the graph
        self._graph.forward()

        # outputs are now changed
        if len(self.__outputs) == 1:
            return self.__outputs[0]
        else:
            return self.__outputs

        

