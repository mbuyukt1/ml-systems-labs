import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def plot_pruning_curves(metrics, output_path):
    frame = pd.DataFrame(metrics)
    frame = frame[frame["experiment"].isin(["global", "layerwise"])]
    if frame.empty:
        return
    plt.figure(figsize=(8, 5))
    for name, group in frame.groupby("experiment"):
        group = group.sort_values("target_sparsity")
        plt.plot(group["target_sparsity"] * 100, group["accuracy"] * 100,
                 marker="o", label=name.capitalize())
    plt.xlabel("Target sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. Unstructured Pruning Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def plot_model_comparison(metrics, output_path):
    frame = pd.DataFrame(metrics)
    if frame.empty:
        return
    frame = frame.drop_duplicates(subset=["model"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(frame["model"], frame["accuracy"] * 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(frame["model"], frame["latency_ms"])
    axes[1].set_ylabel("Latency (ms, batch size 1)")
    axes[1].set_title("Inference latency")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
