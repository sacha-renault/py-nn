from typing import Any
from collections.abc import Callable
from ..tensor import Tensor
from ..graph import ComputeGraph
from ..utils import wrap_as_list
from ..optimizers.optimizer import Optimizer
from .. import xp

class Model:
    def __init__(self, inputs: Tensor | list[Tensor], outputs: Tensor | list[Tensor]) -> None:
        # Ensure type for inputs
        self.__inputs = wrap_as_list(inputs, Tensor)

        # Ensure type for outputs
        self.__outputs = wrap_as_list(outputs, Tensor)

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