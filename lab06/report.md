# Lab 06: Training Pipelines

**Student:** Miray Büyüktepe  
**Student ID:** 73681

## Aim
Build a reproducible YAML-configured training pipeline, measure data loading and training trade-offs, compare optimizers and schedulers, and test gradient accumulation and checkpoint recovery.

## Model and data
The baseline is my best CNN model from Lab 03: 16- and 32-channel convolutions on Fashion-MNIST, normalized with mean 0.2860 and standard deviation 0.3530, using a seeded 55,000/5,000 train-validation split (`seed: 66` in `config.yaml`). Test accuracy is evaluated once on the held-out test split for the initial baseline model.

## Reproduce
Run from the repository root. The existing root requirements include PyTorch, torchvision and MLflow. PyYAML is also required. Python 3.9–3.11 is recommended for the repository's pinned versions.

```bash
pip install -r requirements.txt pyyaml
python lab06/src/run_lab06.py
mlflow ui --backend-store-uri "$(pwd)/lab06/results/mlruns"
```

The first run downloads Fashion-MNIST. GPU is recommended; this is a substantial experiment (more than 70 training epochs plus profiling). Results are hardware-dependent. The script creates CSVs, plots, PyTorch checkpoints, a profiler trace, histories, and MLflow runs under `lab06/results/`. The trace, downloaded data, and MLflow records can be large; consider Git LFS or other storage when sharing them.

## Results
**Pending execution.** Run the script and replace this paragraph with the observed profiling shares, worker bottleneck, batch-size throughput/memory/accuracy trade-offs, optimizer/scheduler comparison, accumulation difference, and interrupted-vs-resumed comparison. No measurements have been invented.

The CPU memory column uses process high-water RSS and may include prior notebook/process allocations; GPU measurements use peak allocated tensor memory. Profiling synchronizes GPU operations and therefore affects absolute timing. The script retains its checkpoint and MLflow history, including a distinct resumed run.
