"""
Dynamic Computational Graph & Reverse-Mode Differentiation Engine
Lead Researcher: Miray Büyüktepe (Student ID: 73681)
Course: Machine Learning Systems (Lab 05)
Institution: Vistula University
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Callable, Optional, Union, List


class ComputationNode:
    """
    Fundamental unit of computation in a dynamic execution graph.
    Stores scalar evaluation, accumulated adjoint gradient, antecedent nodes,
    and a local gradient backward closure.
    """
    __slots__ = ('data', 'gradient', '_inputs', '_backward_hook', 'primitive_op', 'label')

    def __init__(
        self,
        data: Union[float, int],
        _inputs: Tuple[ComputationNode, ...] = (),
        primitive_op: str = '',
        label: Optional[str] = None
    ) -> None:
        self.data: float = float(data)
        self.gradient: float = 0.0
        self._inputs: Set[ComputationNode] = set(_inputs)
        self._backward_hook: Callable[[], None] = lambda: None
        self.primitive_op: str = primitive_op
        self.label: Optional[str] = label

    def __repr__(self) -> str:
        tag = f":{self.label}" if self.label else ""
        return f"ComputationNode({self.data:.4f}, grad={self.gradient:.4f}{tag})"

    def __add__(self, other: Union[ComputationNode, float, int]) -> ComputationNode:
        rhs = other if isinstance(other, ComputationNode) else ComputationNode(other)
        out = ComputationNode(self.data + rhs.data, (self, rhs), 'Add')

        def _backward() -> None:
            self.gradient += 1.0 * out.gradient
            rhs.gradient += 1.0 * out.gradient

        out._backward_hook = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> ComputationNode:
        return self.__add__(other)

    def __mul__(self, other: Union[ComputationNode, float, int]) -> ComputationNode:
        rhs = other if isinstance(other, ComputationNode) else ComputationNode(other)
        out = ComputationNode(self.data * rhs.data, (self, rhs), 'Mul')

        def _backward() -> None:
            self.gradient += rhs.data * out.gradient
            rhs.gradient += self.data * out.gradient

        out._backward_hook = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> ComputationNode:
        return self.__mul__(other)

    def __pow__(self, p: Union[float, int]) -> ComputationNode:
        assert isinstance(p, (int, float)), "Power exponent must be numeric"
        out = ComputationNode(self.data ** p, (self,), f'Pow({p})')

        def _backward() -> None:
            self.gradient += (p * (self.data ** (p - 1))) * out.gradient

        out._backward_hook = _backward
        return out

    def relu(self) -> ComputationNode:
        res = self.data if self.data > 0.0 else 0.0
        out = ComputationNode(res, (self,), 'ReLU')

        def _backward() -> None:
            self.gradient += (1.0 if self.data > 0.0 else 0.0) * out.gradient

        out._backward_hook = _backward
        return out

    def sigmoid(self) -> ComputationNode:
        if self.data >= 0.0:
            val = 1.0 / (1.0 + math.exp(-self.data))
        else:
            val = math.exp(self.data) / (1.0 + math.exp(self.data))
        out = ComputationNode(val, (self,), 'Sigmoid')

        def _backward() -> None:
            self.gradient += (val * (1.0 - val)) * out.gradient

        out._backward_hook = _backward
        return out

    def log(self, eps: float = 1e-12) -> ComputationNode:
        val = max(self.data, eps)
        out = ComputationNode(math.log(val), (self,), 'Log')

        def _backward() -> None:
            self.gradient += (1.0 / val) * out.gradient

        out._backward_hook = _backward
        return out

    def __neg__(self) -> ComputationNode:
        return self * -1.0

    def __sub__(self, other: Union[ComputationNode, float, int]) -> ComputationNode:
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> ComputationNode:
        return ComputationNode(other) + (-self)

    def __truediv__(self, other: Union[ComputationNode, float, int]) -> ComputationNode:
        rhs = other if isinstance(other, ComputationNode) else ComputationNode(other)
        return self * (rhs ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation:
        - Constructs topological DAG ordering via post-order depth-first traversal.
        - Seeds root node gradient to 1.0.
        - Recursively executes backward hooks in reverse order.
        - Preserves gradient accumulation using +=.
        """
        topo: List[ComputationNode] = []
        visited: Set[ComputationNode] = set()

        def build_order(node: ComputationNode) -> None:
            if node not in visited:
                visited.add(node)
                for antecedent in node._inputs:
                    build_order(antecedent)
                topo.append(node)

        build_order(self)
        self.gradient = 1.0

        for n in reversed(topo):
            n._backward_hook()
