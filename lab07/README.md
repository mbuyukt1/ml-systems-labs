# Lab 7 — Compression: Pruning and Knowledge Distillation

A reproducible PyTorch project for the Fashion-MNIST compression lab. It trains an oversized CNN teacher, evaluates global and layer-wise magnitude pruning, applies structured filter pruning, measures model size and inference latency, and trains a compact student with and without knowledge distillation.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- NumPy and Matplotlib

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run the complete experiment (downloads Fashion-MNIST automatically on first use):

```bash
python main.py --data-dir ./data --output-dir ./outputs
```

For a quick smoke test:

```bash
python main.py --epochs 1 --baseline-epochs 1 --fine-tune-epochs 1 --max-train-samples 2000 --max-test-samples 1000
```

The default run uses 5 baseline epochs, 3 fine-tuning epochs per pruning configuration, and 8 student epochs. Training time depends on hardware. Use `--device cpu` or `--device cuda` to select a device explicitly.

## Project structure

```text
lab7_pruning_distillation/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── pruning.py
│   ├── distillation.py
│   ├── evaluation.py
│   └── plots.py
└── outputs/                 # Generated results are written here
```

## Experiments implemented

1. **Baseline:** an oversized CNN teacher (approximately one million parameters) trained on Fashion-MNIST.
2. **Unstructured magnitude pruning:** global and layer-wise pruning at sparsity levels 0%, 25%, 50%, 75%, 90%, and 95%. The smallest-magnitude weights are masked, then the model is fine-tuned while the masks remain fixed.
3. **Structured pruning:** convolutional filters are ranked by L2 norm and whole filters are masked at 10%, 25%, and 50%.
4. **Latency and storage:** dense checkpoint size, sparse COO weight storage estimate, and batch-1 CPU/GPU inference latency are reported separately. Sparse storage savings do not imply faster dense-kernel execution.
5. **Knowledge distillation:** a compact student (approximately 100k parameters) is trained both with ordinary cross-entropy and with a teacher. The distillation settings are `T=4` and `alpha=0.5`.
6. **Efficiency summary:** a CSV table organizes observations under the data, model, compute, and energy pillars.

## Outputs

The script creates the following files in `outputs/`:

- `metrics.csv` — experiment metrics
- `efficiency_summary.csv` — Chapter 9 efficiency-pillar summary
- `pruning_accuracy.png` — accuracy versus sparsity curves
- `model_comparison.png` — accuracy, parameter count, size, and latency comparison
- model checkpoints (`.pth`)

## Important methodology notes

- Pruning masks are applied to weights and remain fixed during fine-tuning. Global pruning selects weights across all eligible layers; layer-wise pruning selects a fraction independently in each layer.
- Structured pruning removes complete filter outputs by applying channel-wise masks. This produces structured sparsity but does not physically rebuild smaller convolution modules. Consequently, the reported dense-kernel latency may show little or no speed-up. Physical channel surgery or sparse-kernel support is required to reliably convert this into hardware acceleration.
- The sparse COO size is an estimate of storing nonzero values and their indices, not a claim that PyTorch's ordinary dense inference kernels will use that representation.
- Energy is not directly measured because reliable energy measurement requires supported hardware counters or an external power meter. The efficiency summary therefore records compute/parameter proxies and explicitly labels energy as not directly measured. Do not present estimates as measured energy.
- Results are generated when you run the project. No accuracy, latency, or energy values are fabricated in this repository.

## Reproducibility

The script sets random seeds and records the selected device and configuration. For a fair latency comparison, run on the same machine, with the same device, warm-up count, and batch size. GPU timings synchronize the device before and after measurements.
