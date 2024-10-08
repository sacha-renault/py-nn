from __future__ import annotations
import numbers

from .. import xp
from ..flags import Flags
from ..operations import (Operation, Multiplication, Addition, Subtraction, Division,
                          Negation)
from ..types import (
    _float16, _float32, _float64,
    _TensorArray,
    ensure_type, ensure_shape, auto_convert_to_cupy
)

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

class Tensor:
    def __init__(self, shape, requires_grad: bool = False) -> None:
        self.__values = xp.zeros(shape, dtype = Flags.global_type())
        self.__grads = xp.zeros(shape, dtype = Flags.global_type())
        self.__requires_grad = requires_grad
        self.__children: list[Tensor] = []
        self._op: Operation = None

    @classmethod
    def zeros(cls, shape, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = xp.zeros(shape, dtype = Flags.global_type())
        return tensor

    @classmethod
    def ones(cls, shape, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = xp.ones(shape, dtype = Flags.global_type())
        return tensor

    @classmethod
    def full(cls, shape, value, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = xp.full(shape, value, dtype = Flags.global_type())
        return tensor

    @classmethod
    def randn(cls, shape, mean = 0, stddev = 1, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = xp.random.normal(mean, stddev, shape) # will be casted by ensure_type
        return tensor

    @classmethod
    def random(cls, shape, min = 0, max = 1, requires_grad: bool = False) -> Tensor:
        tensor = cls(shape, requires_grad)
        tensor.values = xp.random.rand(*shape) * (max - min) + min
        return tensor

    @classmethod
    @auto_convert_to_cupy
    def from_values(cls, values: _TensorArray, requires_grad: bool = False) -> Tensor:
        tensor = cls(values.shape, requires_grad)
        tensor.values = values
        return tensor

    @property
    def children(self) -> list[Tensor]:
        return self.__children

    @property
    def requires_grad(self) -> bool:
        return self.__requires_grad

    def retains_grad(self) -> None:
        self.__requires_grad = True

    @property
    def shape(self):
        return self.__values.shape # we could use grad but it's the same in the end

    @property
    def values(self) -> _TensorArray:
        return self.__values

    @values.setter
    @ensure_type
    @ensure_shape
    def values(self, other: _TensorArray) -> None:
        if isinstance(other, (numbers.Number, xp.generic)):
            other = xp.array([other]) # convert into (1,)
        elif not isinstance(other, xp.ndarray):
            raise TypeError("other must be an array")
        self.__values = other

    @property
    def grads(self) -> _TensorArray:
        return self.__grads

    @grads.setter
    @ensure_type
    @ensure_shape
    def grads(self, other) -> None:
        if not isinstance(other, xp.ndarray):
            raise TypeError("other must be an array")
        self.__grads = other

    @ensure_type
    def accumulate_grads(self, grad_updates: _TensorArray) -> None:
        if self.requires_grad and not Flags.no_grad():
            self.grads += grad_updates

    def reshape(self, new_shape) -> Tensor:
        reshaped = Tensor(new_shape, requires_grad=self.requires_grad)
        reshaped.values = self.values.reshape(new_shape)
        reshaped.grads = self.grads.reshape(new_shape)
        reshaped.add_children(self)
        # reshaped.set_operation(Reshape(self.shape, new_shape))
        return reshaped

    def _zero_grad(self) -> None:
        if self.grads is not None:
            self.grads.fill(0)

    def zero_grad(self) -> None:
        for node in _topological_order(self):
            xp.copyto(node.grads, xp.zeros_like(node.grads))

    def set_operation(self, operation: Operation) -> None:
        if (isinstance(operation, type) and issubclass(operation, Operation) or
            isinstance(operation, Operation)):
            self._op = operation
        else:
            raise TypeError(f"operation must be Operation type, not {type(operation)}")

    def add_children(self, *children) -> None:
        for child in children:
            self.children.append(child)

    def forward(self) -> None:
        if self._op is not None:
            result_values = self._op.forward(*(child.values for child in self.children))
            xp.copyto(self.values, result_values)
            # self.values = result_values

    def backward(self) -> None:
        xp.copyto(self.grads, xp.ones_like(self.grads))
        for node in reversed(_topological_order(self)):
            node._backward()

    def _backward(self) -> None:
        if self._op is not None:
            children_grads_update = self._op.backward(
                self.grads,
                self.values,
                *(child.values for child in self.children)
            )

            for child, grads_updates in zip(self.children, children_grads_update):
                child.accumulate_grads(grads_updates) # update grads for every child

    # OPERATIONS
    def __mul__(self, other: Tensor) -> Tensor:
        result = Multiplication.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.requires_grad or other.requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Multiplication)
        return tensor

    def __add__(self, other: Tensor) -> Tensor:
        result = Addition.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.requires_grad or other.requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Addition)
        return tensor

    def __sub__(self, other: Tensor) -> Tensor:
        result = Subtraction.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.requires_grad or other.requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Subtraction)
        return tensor

    def __truediv__(self, other: Tensor) -> Tensor:
        result = Division.forward(self.values, other.values)
        tensor = Tensor.from_values(result, requires_grad=self.requires_grad or other.requires_grad)
        tensor.add_children(self, other)
        tensor.set_operation(Division)
        return tensor

    def __neg__(self) -> Tensor:
        result = -self.values
        tensor = Tensor.from_values(result, requires_grad=self.requires_grad)
        tensor.add_children(self)
        tensor.set_operation(Negation)
        return tensor

    def __getitem__(self, key) -> Tensor:
        # to avoid circular import, TensorView is imported here
        from .tensor_view import _TensorView
        tensor = _TensorView(self, key)
        # tensor.add_children(self)
        return tensor

    def __repr__(self) -> str:
        return f"<Tensor: shape={self.shape}>"
