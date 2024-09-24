from ..tensor import Tensor
from ..tensor.tensor import _topological_order
from ..tensor.sub_class import WeightTensor, BiasTensor
from .. import xp


class ComputeGraph:
    def __init__(self, model_output: Tensor | list[Tensor]) -> None:
        if isinstance(model_output, list):
            virtual_output = Tensor((1))
            virtual_output.add_children(*model_output)
            ordered_nodes = _topological_order(virtual_output)[:-1]
        else:
            ordered_nodes = _topological_order(model_output)

        self.__ordered_nodes = ordered_nodes

    def forward(self) -> None:
        for node in self.__ordered_nodes:
            node.forward()

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