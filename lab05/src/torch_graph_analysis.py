"""
PyTorch Execution Graph Tracing & Contiguity Latency Analysis
Lead Researcher: Miray Büyüktepe (Student ID: 73681)
"""

import time
import json
import os
import torch
import torch.nn as nn


class ExperimentalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc2(self.relu(self.fc1(x))))


def profile_all():
    print("[MIRAY BUYUKTEPE] Profiling PyTorch eager vs traced & memory contiguity...")
    model = ExperimentalNet().eval()
    inp = torch.randn(64, 2, dtype=torch.float32)

    traced_model = torch.jit.trace(model, inp)
    graph_ir = str(traced_model.inlined_graph)

    # Warm-up
    for _ in range(120):
        _ = model(inp)
        _ = traced_model(inp)

    # Timing eager vs traced
    loops = 1000
    t0 = time.perf_counter()
    for _ in range(loops):
        _ = model(inp)
    eager_us = ((time.perf_counter() - t0) / loops) * 1e6

    t0 = time.perf_counter()
    for _ in range(loops):
        _ = traced_model(inp)
    traced_us = ((time.perf_counter() - t0) / loops) * 1e6

    print(f"  Eager Mode:  {eager_us:.2f} µs")
    print(f"  Traced Mode: {traced_us:.2f} µs (Speedup: {eager_us / traced_us:.2f}x)")

    # Tensor strides & memory contiguity
    DIM = 2048
    T_contiguous = torch.randn(DIM, DIM, dtype=torch.float32)
    T_transposed = T_contiguous.t()

    c_meta = {
        'shape': list(T_contiguous.shape),
        'stride': list(T_contiguous.stride()),
        'is_contiguous': T_contiguous.is_contiguous()
    }
    nc_meta = {
        'shape': list(T_transposed.shape),
        'stride': list(T_transposed.stride()),
        'is_contiguous': T_transposed.is_contiguous()
    }

    reps = 200
    for _ in range(20):
        _ = torch.sum(T_contiguous, dim=1)
        _ = torch.sum(T_transposed, dim=1)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(T_contiguous, dim=1)
    t_c = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(T_transposed, dim=1)
    t_nc = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = T_transposed.contiguous()
    t_copy = ((time.perf_counter() - t0) / reps) * 1000.0

    slowdown = t_nc / t_c
    print(f"  Contiguous Sum:     {t_c:.3f} ms")
    print(f"  Non-Contiguous Sum: {t_nc:.3f} ms")
    print(f"  Penalty Factor:     {slowdown:.2f}x")
    print(f"  Buffer Clone Cost:  {t_copy:.3f} ms")

    data = {
        'student': 'Miray Büyüktepe (73681)',
        'eager_us': eager_us,
        'traced_us': traced_us,
        'speedup': eager_us / traced_us,
        'contiguous_info': c_meta,
        'non_contiguous_info': nc_meta,
        'time_contiguous_ms': t_c,
        'time_non_contiguous_ms': t_nc,
        'slowdown_factor': slowdown,
        'reallocation_ms': t_copy,
        'ir_graph_sample': graph_ir[:600]
    }

    os.makedirs('results', exist_ok=True)
    with open('results/strides_profiling.json', 'w') as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == '__main__':
    profile_all()
