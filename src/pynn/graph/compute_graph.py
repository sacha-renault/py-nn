from ..tensor import Tensor
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
            output.grads = xp.ones(output.shape)
            
        for node in reversed(self.__ordered_nodes):
            node.backward()