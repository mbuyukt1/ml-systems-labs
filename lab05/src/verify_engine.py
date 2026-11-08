"""
Gradient Verification Suite: Custom Engine vs. PyTorch Autograd & Central Differences
Lead Researcher: Miray Büyüktepe (Student ID: 73681)
"""

import json
import os
import torch
from scalar_math import ComputationNode


def evaluate_graph(a_list, b_list, c_list):
    terms = []
    for a, b, c in zip(a_list, b_list, c_list):
        t = ((a * b + c) ** 2).relu()
        terms.append(t)
    
    total = terms[0]
    for n in terms[1:]:
        total = total + n
    return total


def finite_diff(a_vals, b_vals, c_vals, param, idx, h=1e-7):
    def f(a, b, c):
        return sum(max(0.0, (ai * bi + ci) ** 2) for ai, bi, ci in zip(a, b, c))

    ap, am = list(a_vals), list(a_vals)
    bp, bm = list(b_vals), list(b_vals)
    cp, cm = list(c_vals), list(c_vals)

    if param == 'a':
        ap[idx] += h
        am[idx] -= h
    elif param == 'b':
        bp[idx] += h
        bm[idx] -= h
    else:
        cp[idx] += h
        cm[idx] -= h

    return (f(ap, bp, cp) - f(am, bm, cm)) / (2.0 * h)


def run_test():
    print("[MIRAY BUYUKTEPE] Verifying automatic differentiation engine...")
    a_in = [1.25, -2.10, 0.45, -1.80]
    b_in = [-2.00, 1.50, 3.20, 0.90]
    c_in = [1.50, -0.80, -1.40, 2.10]

    # Custom ComputationNode Pass
    a_nodes = [ComputationNode(x, label=f'a_{i}') for i, x in enumerate(a_in)]
    b_nodes = [ComputationNode(x, label=f'b_{i}') for i, x in enumerate(b_in)]
    c_nodes = [ComputationNode(x, label=f'c_{i}') for i, x in enumerate(c_in)]

    total_loss = evaluate_graph(a_nodes, b_nodes, c_nodes)
    total_loss.backward()

    # PyTorch Reference Pass
    ta = torch.tensor(a_in, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_in, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c_in, dtype=torch.float64, requires_grad=True)

    t_loss = (((ta * tb + tc) ** 2).relu()).sum()
    t_loss.backward()

    # Numerical Central Finite Differences
    max_torch_diff = 0.0
    max_fd_diff = 0.0
    records = []

    for name, nodes, t_ref in [('a', a_nodes, ta), ('b', b_nodes, tb), ('c', c_nodes, tc)]:
        for i in range(len(a_in)):
            g_cust = nodes[i].gradient
            g_torch = t_ref.grad[i].item()
            g_fd = finite_diff(a_in, b_in, c_in, name, i)

            diff_torch = abs(g_cust - g_torch)
            diff_fd = abs(g_cust - g_fd)

            max_torch_diff = max(max_torch_diff, diff_torch)
            max_fd_diff = max(max_fd_diff, diff_fd)

            records.append({
                'variable': f"{name}_{i}",
                'custom_gradient': g_cust,
                'pytorch_gradient': g_torch,
                'numerical_gradient': g_fd,
                'diff_torch': diff_torch,
                'diff_fd': diff_fd
            })

    passed = bool(max_torch_diff < 1e-6)
    print(f"  Max Diff vs PyTorch: {max_torch_diff:.2e}")
    print(f"  Max Diff vs Finite Diff: {max_fd_diff:.2e}")
    print(f"  Status: {'PASSED (< 1e-6)' if passed else 'FAILED'}")

    res = {
        'student': 'Miray Büyüktepe (73681)',
        'expression': '((a * b + c) ** 2).relu().sum()',
        'loss_custom': total_loss.data,
        'loss_pytorch': t_loss.item(),
        'max_abs_diff_torch': max_torch_diff,
        'max_abs_diff_fd': max_fd_diff,
        'threshold': 1e-6,
        'passed': passed,
        'audit': records
    }

    os.makedirs('results', exist_ok=True)
    with open('results/engine_validation.json', 'w') as f:
        json.dump(res, f, indent=2)

    return res


if __name__ == '__main__':
    run_test()
