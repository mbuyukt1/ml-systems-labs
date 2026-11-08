"""
Empirical Training Benchmark on make_moons: Custom Engine vs. PyTorch Baseline
Lead Researcher: Miray Büyüktepe (Student ID: 73681)
Network Architecture: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE loss)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from scalar_math import ComputationNode


class CompactNeuron:
    def __init__(self, n_in: int, is_relu: bool = True):
        # He normal initialization
        std = (2.0 / n_in) ** 0.5
        self.w = [ComputationNode(random.gauss(0.0, std)) for _ in range(n_in)]
        self.b = ComputationNode(0.0)
        self.is_relu = is_relu

    def forward(self, x):
        acc = self.b
        for wi, xi in zip(self.w, x):
            acc = acc + (wi * xi)
        return acc.relu() if self.is_relu else acc

    def params(self):
        return self.w + [self.b]


class DenseLayer:
    def __init__(self, n_in: int, n_out: int, is_relu: bool = True):
        self.neurons = [CompactNeuron(n_in, is_relu=is_relu) for _ in range(n_out)]

    def forward(self, x):
        res = [n.forward(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res

    def params(self):
        p = []
        for n in self.neurons:
            p.extend(n.params())
        return p


class MoonsMLP:
    def __init__(self):
        self.layer1 = DenseLayer(2, 8, is_relu=True)
        self.layer2 = DenseLayer(8, 1, is_relu=False)

    def forward(self, x):
        h = self.layer1.forward(x)
        out = self.layer2.forward(h)
        return out.sigmoid()

    def params(self):
        return self.layer1.params() + self.layer2.params()

    def zero_grad(self):
        for p in self.params():
            p.gradient = 0.0


class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def bce_criterion(y_true, y_pred):
    n = len(y_true)
    losses = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            term = -yp.log()
        else:
            term = -(ComputationNode(1.0) - yp).log()
        losses.append(term)
    
    total = losses[0]
    for l in losses[1:]:
        total = total + l
    return total * (1.0 / n)


def train_custom(X, y, epochs=35, lr=0.25):
    random.seed(73681)
    model = MoonsMLP()
    X_nodes = [[ComputationNode(v) for v in row] for row in X]
    
    t0 = time.perf_counter()
    loss_hist = []
    
    for epoch in range(epochs):
        preds = [model.forward(x) for x in X_nodes]
        loss = bce_criterion(y, preds)
        
        model.zero_grad()
        loss.backward()
        
        for p in model.params():
            p.data -= lr * p.gradient
            
        loss_hist.append(loss.data)
        
    duration = time.perf_counter() - t0
    acc = sum((1 if p.data >= 0.5 else 0) == yt for p, yt in zip(preds, y)) / len(y)
    return duration, loss_hist[-1], acc


def train_pytorch(X, y, epochs=35, lr=0.25):
    torch.manual_seed(73681)
    model = PyTorchModel()
    crit = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    loss_hist = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(tx)
        loss = crit(out, ty)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        
    duration = time.perf_counter() - t0
    with torch.no_grad():
        preds = (model(tx) >= 0.5).float()
        acc = (preds == ty).float().mean().item()
    return duration, loss_hist[-1], acc


def run_benchmark():
    print("[MIRAY BUYUKTEPE] Executing 5-run make_moons benchmark...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    c_times = []
    p_times = []
    c_losses = []
    p_losses = []
    
    for trial in range(5):
        tc, lc, ac = train_custom(X, y, epochs=35, lr=0.25)
        tp, lp, ap = train_pytorch(X, y, epochs=35, lr=0.25)
        
        c_times.append(tc * 1000.0)
        p_times.append(tp * 1000.0)
        c_losses.append(lc)
        p_losses.append(lp)
        print(f"  Trial {trial+1} | Custom: {c_times[-1]:.2f} ms | PyTorch: {p_times[-1]:.2f} ms")
        
    med_custom = statistics.median(c_times)
    med_pytorch = statistics.median(p_times)
    speedup = med_custom / med_pytorch
    
    telemetry = {
        'student': 'Miray Büyüktepe (73681)',
        'trials': 5,
        'custom_median_ms': med_custom,
        'pytorch_median_ms': med_pytorch,
        'slowdown_factor': speedup,
        'custom_losses': c_losses,
        'pytorch_losses': p_losses,
        'custom_times_ms': c_times,
        'pytorch_times_ms': p_times
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_5runs_telemetry.json', 'w') as f:
        json.dump(telemetry, f, indent=2)
        
    print(f"\n[SUMMARY] Custom Median: {med_custom:.2f} ms | PyTorch Median: {med_pytorch:.2f} ms | Ratio: {speedup:.1f}x")
    return telemetry


if __name__ == '__main__':
    run_benchmark()
