import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models import BaselineCNN, StudentCNN, count_parameters
from src.pruning import (apply_magnitude_pruning, apply_structured_filter_pruning,
                         enforce_pruning_masks, overall_sparsity)
from src.distillation import train_student
from src.evaluation import evaluate, measure_latency_ms, serialized_size_mb, sparse_coo_size_mb
from src.plots import plot_pruning_curves, plot_model_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST pruning and distillation experiments")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--student-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Optional cap for quick tests; 0 uses the full training set.")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="Optional cap for quick tests; 0 uses the full test set.")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.max_train_samples:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    if args.max_test_samples:
        test_set = Subset(test_set, range(min(args.max_test_samples, len(test_set))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def train_classifier(model, loader, device, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / max(seen, 1):.4f}")

def record_model(name, experiment, model, test_loader, device, target_sparsity=0.0):
    result = evaluate(model, test_loader, device)
    row = {
        "model": name, "experiment": experiment, "target_sparsity": target_sparsity,
        "actual_sparsity": overall_sparsity(model),
        "accuracy": result["accuracy"], "loss": result["loss"],
        "parameters": count_parameters(model),
        "dense_size_mb": serialized_size_mb(model),
        "sparse_coo_estimate_mb": sparse_coo_size_mb(model),
        "latency_ms": measure_latency_ms(model, device),
        "device": str(device),
    }
    print(f"{name}: accuracy={row['accuracy']:.4f}, params={row['parameters']:,}, "
          f"latency={row['latency_ms']:.3f} ms")
    return row

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args)
    rows = []

    # 1. Train the oversized baseline teacher.
    teacher = BaselineCNN().to(device)
    print(f"Baseline trainable parameters: {count_parameters(teacher):,}")
    train_classifier(teacher, train_loader, device, args.baseline_epochs, args.learning_rate)
    rows.append(record_model("baseline_teacher", "baseline", teacher, test_loader, device))
    torch.save(teacher.state_dict(), output_dir / "baseline_teacher.pth")

    # 2. Global and layer-wise magnitude pruning, with fixed masks during fine-tuning.
    sparsity_targets = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    for mode in ["global", "layerwise"]:
        for amount in sparsity_targets:
            model = copy.deepcopy(teacher)
            apply_magnitude_pruning(model, amount, mode=mode)
            if args.fine_tune_epochs > 0 and amount > 0:
                # Pruning reparameterizations keep masks fixed through optimizer updates.
                train_classifier(model, train_loader, device, args.fine_tune_epochs, args.learning_rate * 0.25)
            enforce_pruning_masks(model)
            row = record_model(f"{mode}_{int(amount * 100)}", mode, model,
                               test_loader, device, amount)
            rows.append(row)
            torch.save(model.state_dict(), output_dir / f"{mode}_{int(amount * 100)}.pth")

    # 3. Structured filter pruning by L2 norm.
    for amount in [0.10, 0.25, 0.50]:
        model = copy.deepcopy(teacher)
        apply_structured_filter_pruning(model, amount)
        if args.fine_tune_epochs > 0:
            train_classifier(model, train_loader, device, args.fine_tune_epochs, args.learning_rate * 0.25)
        enforce_pruning_masks(model)
        row = record_model(f"structured_{int(amount * 100)}", "structured",
                           model, test_loader, device, amount)
        rows.append(row)
        torch.save(model.state_dict(), output_dir / f"structured_{int(amount * 100)}.pth")

    # 4. Train the compact student from scratch, with and without the teacher.
    plain_student = StudentCNN().to(device)
    train_student(plain_student, None, train_loader, device, args.student_epochs,
                  args.learning_rate, use_distillation=False)
    rows.append(record_model("student_plain", "student", plain_student, test_loader, device))
    torch.save(plain_student.state_dict(), output_dir / "student_plain.pth")

    distilled_student = StudentCNN().to(device)
    train_student(distilled_student, teacher, train_loader, device, args.student_epochs,
                  args.learning_rate, use_distillation=True, temperature=4.0, alpha=0.5)
    rows.append(record_model("student_distilled", "distillation", distilled_student,
                             test_loader, device))
    torch.save(distilled_student.state_dict(), output_dir / "student_distilled.pth")

    # Save all measurements and plots.
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    plot_pruning_curves(rows, output_dir / "pruning_accuracy.png")
    plot_model_comparison(rows, output_dir / "model_comparison.png")

    summary = pd.DataFrame([
        {"efficiency_pillar": "Data", "evidence": "Fashion-MNIST; same train/test split and preprocessing for all models."},
        {"efficiency_pillar": "Model", "evidence": "Compare baseline, global/layer-wise/structured pruning, plain student, and distilled student using parameter count, sparsity, and checkpoint size."},
        {"efficiency_pillar": "Compute", "evidence": "Report measured batch-1 latency on the selected device; dense kernels may not benefit from unstructured zero weights."},
        {"efficiency_pillar": "Energy", "evidence": "Not directly measured. Parameter count and compute/latency are proxies only; hardware power measurement is required for energy claims."},
    ])
    summary.to_csv(output_dir / "efficiency_summary.csv", index=False)
    print(f"\nFinished. Results saved to: {output_dir.resolve()}")
    print("Review metrics.csv and efficiency_summary.csv before writing the lab conclusion.")

if __name__ == "__main__":
    main()
