# Lab 5: Frameworks and Automatic Differentiation

This folder contains the scalar reverse-mode autodiff engine, the runnable experiments, and their recorded results.

## Run

From the repository root, install requirements.txt, then run:

    python lab5/src/run_experiments.py

The script writes lab5/results/lab5_results.json. It validates the custom gradients against PyTorch autograd and central finite differences, compares five-run median training times on a seeded two-moons dataset, traces a small PyTorch model, and measures matrix multiplication with contiguous and strided tensors.

The two-moons points are generated from the standard interleaving half-circle construction with NumPy so that scikit-learn is not needed.
