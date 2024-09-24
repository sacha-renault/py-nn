from ..tensor import Tensor
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp

def _topological_order(output_tensor: Tensor) -> list[Tensor]:
    seen = set()
    ordered_nodes: list[Tensor] = []

    def build_graph(node: Tensor):
        if node in seen:
            return
        seen.add(node)  # Add the node to the seen set
        for child in node.children:
            build_graph(child)
        ordered_nodes.append(node)

    build_graph(output_tensor)
    return ordered_nodes


class ComputeGraph:
    def __init__(self, model_output: Tensor | list[Tensor]) -> None:
        if isinstance(model_output, list):
            virtual_output = Tensor((1))
            virtual_output.add_children(*model_output)
            ordered_nodes = _topological_order(virtual_output)[:-1]
            self.__outputs = model_output
        else:
            ordered_nodes = _topological_order(model_output)
            self.__outputs = [model_output]

        self.__ordered_nodes = ordered_nodes

    def forward(self) -> None:
        for node in self.__ordered_nodes:
            node.forward()

    def backward(self) -> None:
        # start by setting grad = 1 for every output
        for output in self.__outputs:
            xp.copyto(output.grads, xp.ones(output.shape))

        for node in reversed(self.__ordered_nodes):
            node.backward()

    def zero_grad(self) -> None:
        for node in self.__ordered_nodes:
            node.zero_grad()

    def is_tensor_in(self, tensor: Tensor) -> bool:
        return tensor in self.__ordered_nodes

    @property
    def parameters(self) -> list[Tensor]:
        params = []
        for node in self.__ordered_nodes:
            if isinstance(node, (WeightTensor, BiasTensor)):
                params.append(node)
        return params

    @property
    def nodes(self) -> list[Tensor]:
        return self.__ordered_nodes

    def __len__(self) -> int:
        return len(self.__ordered_nodes)