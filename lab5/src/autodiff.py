"""A small scalar reverse-mode automatic differentiation engine."""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value and the local graph needed for reverse-mode AD."""

    __slots__ = ("data", "grad", "_parents", "_backward", "_op")

    def __init__(
        self,
        data: float,
        parents: Iterable["Value"] = (),
        backward: Callable[[], None] | None = None,
        op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)
        self._backward = backward or (lambda: None)
        self._op = op

    def __add__(self, other: float | "Value") -> "Value":
        other = as_value(other)
        out = Value(self.data + other.data, (self, other), op="+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-as_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return as_value(other) + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = as_value(other)
        out = Value(self.data * other.data, (self, other), op="*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, exponent: float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Value powers must be real constants")
        out = Value(self.data**exponent, (self,), op=f"**{exponent}")

        def backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), op="ReLU")

        def backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = backward
        return out

    def sigmoid(self) -> "Value":
        # This stable form avoids overflow for large negative inputs.
        if self.data >= 0:
            z = math.exp(-self.data)
            result = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            result = z / (1.0 + z)
        out = Value(result, (self,), op="sigmoid")

        def backward() -> None:
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError("log is defined only for positive values")
        out = Value(math.log(self.data), (self,), op="log")

        def backward() -> None:
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def backward(self) -> None:
        """Backpropagate from this scalar, clearing stale graph gradients."""
        topo: list[Value] = []
        visited: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            topo.append(node)

        visit(self)
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


def as_value(value: float | Value) -> Value:
    return value if isinstance(value, Value) else Value(value)


def value_sum(values: Iterable[Value]) -> Value:
    result = Value(0.0)
    for value in values:
        result = result + value
    return result
