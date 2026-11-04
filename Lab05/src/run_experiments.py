"""Run Lab 5 gradient, training, and PyTorch execution measurements."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def scalar_expression(a, b, c):
    return ((a * b + c) ** 2).relu()


def verify_gradients() -> dict:
    # Several independent elements are summed to exercise gradient accumulation.
    av = np.array([-1.2, -0.4, 0.1, 0.7, 1.3, 0.2, -0.8, 0.9], dtype=np.float64)
    bv = np.array([0.5, -1.1, 0.8, 0.3, -0.6, 1.2, 0.4, -0.7], dtype=np.float64)
    cv = np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.8, 0.2], dtype=np.float64)
    a = [Value(x) for x in av]
    b = [Value(x) for x in bv]
    c = [Value(x) for x in cv]
    loss = value_sum(scalar_expression(x, y, z) for x, y, z in zip(a, b, c))
    loss.backward()
    custom = np.array([[x.grad for x in group] for group in (a, b, c)])

    ta = torch.tensor(av, requires_grad=True)
    tb = torch.tensor(bv, requires_grad=True)
    tc = torch.tensor(cv, requires_grad=True)
    torch_loss = torch.relu((ta * tb + tc) ** 2).sum()
    torch_loss.backward()
    torch_grad = np.stack([ta.grad.numpy(), tb.grad.numpy(), tc.grad.numpy()])

    def numeric(values: np.ndarray, index: int, which: int, eps: float = 1e-6) -> float:
        args = [av.copy(), bv.copy(), cv.copy()]
        plus, minus = [x.copy() for x in args], [x.copy() for x in args]
        plus[which][index] += eps
        minus[which][index] -= eps

        def objective(parts):
            aa, bb, cc = parts
            return float(np.maximum((aa * bb + cc) ** 2, 0.0).sum())

        return (objective(plus) - objective(minus)) / (2 * eps)

    numeric_grad = np.array(
        [[numeric(group, i, j) for i in range(len(av))] for j, group in enumerate((av, bv, cv))]
    )
    return {
        "torch_max_abs_difference": float(np.max(np.abs(custom - torch_grad))),
        "finite_difference_max_abs_difference": float(np.max(np.abs(custom - numeric_grad))),
        "tolerance": 1e-6,
        "passed": bool(
            np.max(np.abs(custom - torch_grad)) < 1e-6
            and np.max(np.abs(custom - numeric_grad)) < 1e-6
        ),
        "elements_per_input": len(av),
        "loss": float(loss.data),
    }


def make_custom_model(seed: int):
    rng = np.random.default_rng(seed)
    w1 = [[Value(float(rng.normal(0, 0.35))) for _ in range(2)] for _ in range(8)]
    b1 = [Value(0.0) for _ in range(8)]
    w2 = [Value(float(rng.normal(0, 0.35))) for _ in range(8)]
    b2 = Value(0.0)
    params = [*sum(w1, []), *b1, *w2, b2]

    def forward(features):
        hidden = [
            (w[0] * float(features[0]) + w[1] * float(features[1]) + bias).relu()
            for w, bias in zip(w1, b1)
        ]
        return (value_sum(weight * h for weight, h in zip(w2, hidden)) + b2).sigmoid()

    return forward, params


def train_custom(x, y, seed: int, epochs: int = 18, lr: float = 0.08) -> float:
    forward, params = make_custom_model(seed)
    for _ in range(epochs):
        for features, label in zip(x, y):
            p = forward(features)
            # Binary cross entropy; clamp keeps logs finite at machine precision.
            q = min(max(p.data, 1e-12), 1 - 1e-12)
            loss = -(float(label) * p.log() + (1.0 - float(label)) * (1.0 - p).log())
            loss.backward()
            for param in params:
                param.data -= lr * param.grad
    return evaluate_custom(forward, x, y)


def evaluate_custom(forward, x, y) -> float:
    correct = 0
    for features, label in zip(x, y):
        correct += int((forward(features).data >= 0.5) == bool(label))
    return correct / len(y)


def train_torch(x, y, seed: int, epochs: int = 18, lr: float = 0.08) -> float:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs):
        for i in range(len(tx)):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tx[i : i + 1]), ty[i : i + 1])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        accuracy = ((model(tx) >= 0.5) == ty.bool()).float().mean().item()
    return accuracy


def median_training_times(x, y) -> dict:
    custom_times, torch_times, scores = [], [], {"custom": [], "pytorch": []}
    for seed in range(5):
        start = time.perf_counter()
        scores["custom"].append(train_custom(x, y, 73681 + seed))
        custom_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        scores["pytorch"].append(train_torch(x, y, 73681 + seed))
        torch_times.append(time.perf_counter() - start)
    return {
        "runs": 5,
        "epochs": 18,
        "samples": len(x),
        "custom_median_seconds": statistics.median(custom_times),
        "pytorch_median_seconds": statistics.median(torch_times),
        "custom_accuracy_per_run": scores["custom"],
        "pytorch_accuracy_per_run": scores["pytorch"],
    }


def generate_moons(n_samples: int = 128, noise: float = 0.15, seed: int = 73681):
    """Generate the standard two interleaving half circles without sklearn."""
    if n_samples % 2:
        raise ValueError("n_samples must be even")
    count = n_samples // 2
    angles = np.linspace(0.0, math.pi, count)
    outer = np.column_stack((np.cos(angles), np.sin(angles)))
    inner = np.column_stack((1.0 - np.cos(angles), 0.5 - np.sin(angles)))
    x = np.vstack((outer, inner))
    y = np.concatenate((np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)))
    rng = np.random.default_rng(seed)
    x += rng.normal(0.0, noise, size=x.shape)
    order = rng.permutation(n_samples)
    return x[order], y[order]


def median_latency(fn, warmup: int = 30, iterations: int = 300, repeats: int = 5) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            samples.append((time.perf_counter() - start) / iterations)
    return statistics.median(samples) * 1e6


def inspect_framework() -> dict:
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).eval()
    example = torch.randn(128, 2)
    traced = torch.jit.trace(model, example)
    eager_us = median_latency(lambda: model(example))
    traced_us = median_latency(lambda: traced(example))

    rows, cols = 256, 256
    base = torch.randn(rows, cols * 2)
    noncontiguous = base[:, ::2]
    contiguous = noncontiguous.contiguous()
    weight = torch.randn(cols, cols)
    contig_matmul_us = median_latency(lambda: contiguous @ weight, iterations=50, repeats=5)
    strided_matmul_us = median_latency(lambda: noncontiguous @ weight, iterations=50, repeats=5)

    return {
        "execution": {
            "method": "torch.jit.trace",
            "warmup_calls": 30,
            "iterations_per_repeat": 300,
            "repeats": 5,
            "eager_median_us_per_call": eager_us,
            "traced_median_us_per_call": traced_us,
            "traced_speedup": eager_us / traced_us if traced_us else None,
            "graph_operators": [node.kind() for node in traced.inlined_graph.nodes()],
        },
        "tensor_layout": {
            "shape": list(noncontiguous.shape),
            "dtype": str(noncontiguous.dtype),
            "contiguous_stride": list(contiguous.stride()),
            "noncontiguous_stride": list(noncontiguous.stride()),
            "contiguous_is_contiguous": contiguous.is_contiguous(),
            "view_is_contiguous": noncontiguous.is_contiguous(),
            "matmul_contiguous_median_us": contig_matmul_us,
            "matmul_view_median_us": strided_matmul_us,
            "view_over_contiguous_ratio": strided_matmul_us / contig_matmul_us,
        },
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    import autodiff

    globals()["Value"] = autodiff.Value
    globals()["value_sum"] = autodiff.value_sum
    x, y = generate_moons(n_samples=128, noise=0.15, seed=73681)
    result = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "gradient_validation": verify_gradients(),
        "training": median_training_times(x, y),
        "framework": inspect_framework(),
    }
    (RESULTS / "lab5_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
