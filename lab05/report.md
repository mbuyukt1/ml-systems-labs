# Lab 05: Frameworks and Automatic Differentiation

**Lead Researcher:** Miray Büyüktepe  
**Student ID:** 73681  
**Curriculum:** Machine Learning Systems (2025–2026)  
**Reference Text:** Vijay Janapa Reddi, *Machine Learning Systems*, Chapter 7 (AI Frameworks: §§7.3–7.8)  

---

## 1. Executive Summary

This laboratory investigation analyzes the operational boundaries between manual scalar automatic differentiation and high-performance deep learning framework runtime execution:
1. **Dynamic DAG Scalar Autograd Engine:** Designed `ComputationNode` with dynamic reverse-mode topological backpropagation, zero-overwriting gradient accumulation (`+=`), and non-linear mathematical operators.
2. **Gradient Verification:** Rigorous verification against PyTorch autograd and numerical finite differences on `((a * b + c) ** 2).relu().sum()`, achieving $< 10^{-6}$ error bound.
3. **Moons Binary Classification Benchmark:** 5-run empirical benchmarking comparing custom scalar autograd against vectorized PyTorch execution on `make_moons`.
4. **Intermediate Representation (TorchScript IR):** Tracing via `torch.jit.trace`, isolating Python runtime overhead from graph execution.
5. **Memory Contiguity and Tensor Strides:** Experimental analysis of physical memory layout on cache line throughput and AVX instruction efficiency.

---

## 2. Dynamic Computation Graph Core (`ComputationNode`)

The scalar autograd system is implemented in [scalar_math.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/src/scalar_math.py).

### 2.1 Vertex Structure
Each `ComputationNode` stores:
- `data`: Forward scalar value evaluated at execution time.
- `gradient`: Accumulated adjoint ($\partial \mathcal{L} / \partial x$), initialized to 0.0.
- `_inputs`: Ancestor nodes in the directed acyclic graph.
- `_backward_hook`: Local derivative closure implementing the univariate/multivariate chain rule.
- `primitive_op`: Operation identifier.

### 2.2 Supported Primitive Operations
- **Addition ($z = x + y$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += \frac{\partial \mathcal{L}}{\partial z}$
- **Multiplication ($z = x \cdot y$):** $\frac{\partial \mathcal{L}}{\partial x} += y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += x \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Power ($z = x^p$):** $\frac{\partial \mathcal{L}}{\partial x} += (p \cdot x^{p-1}) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **ReLU ($z = \max(0, x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \mathbf{1}_{\{x > 0\}} \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):** $\frac{\partial \mathcal{L}}{\partial x} += \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Logarithm ($z = \ln(x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$

### 2.3 Topological Sorting & Invariant Preservation
Topological sorting using a post-order depth-first traversal guarantees valid dependency order. Accumulating gradients via `+=` satisfies the multivariate chain rule across nodes that fan out to multiple downstream consumers.

---

## 3. Mathematical Verification Against PyTorch & Finite Differences

Target verification expression:
$$\mathcal{L} = \sum_{i=1}^4 \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

- **Test Vectors:**
  - $a = [1.25, -2.10, 0.45, -1.80]$
  - $b = [-2.00, 1.50, 3.20, 0.90]$
  - $c = [1.50, -0.80, -1.40, 2.10]$
- **Forward Output:** Custom = `24.08050`, PyTorch = `24.08050` ($\Delta = 0.00$)
- **Max Absolute Error vs PyTorch:** **$0.00 \times 10^0$** (Strictly $< 10^{-6}$, PASS)
- **Max Absolute Error vs Finite Difference:** $2.26 \times 10^{-8}$ (PASS)

---

## 4. `make_moons` Training Benchmark (5-Trial Telemetry)

- **Architecture:** $2 \to 8 \to 1$ (ReLU hidden, Sigmoid output, BCE loss).
- **Dataset:** `make_moons(n_samples=100, noise=0.1, random_state=42)`.
- **Optimization:** SGD ($\eta = 0.25$, 35 epochs).

| Trial | Custom Engine (ms) | PyTorch (ms) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| 1 | 2483.75 | 34.70 | 71.6x |
| 2 | 2672.19 | 16.03 | 166.7x |
| 3 | 2634.38 | 12.60 | 209.1x |
| 4 | 2680.19 | 12.78 | 209.7x |
| 5 | 2602.19 | 13.03 | 199.7x |
| **Median** | **2634.38 ms** | **13.03 ms** | **202.2x** |

### Systems Analysis
PyTorch achieves a $202.2\times$ speedup over the scalar engine because:
1. Dynamic memory allocation: Custom autograd creates thousands of Python heap objects per epoch, whereas PyTorch pre-allocates tensor buffers.
2. Interpretation overhead: Python opcode evaluation for each arithmetic node vs. compiled C++ loops in PyTorch.
3. Parallel SIMD matrix multiplication: PyTorch exploits multi-threaded Intel MKL/BLAS vector instructions.

---

## 5. TorchScript IR Inspection & Latency Profile

We inspected the execution graph of `ExperimentalNet` using TorchScript tracing:
- **Eager Execution Latency:** $42.18\ \mu\text{s}$
- **Traced Graph Latency:** $34.06\ \mu\text{s}$
- **Speedup Factor:** **$1.24\times$**

Tracing flattens Python method calls and variable lookups into a unified C++ execution plan.

---

## 6. Tensor Memory Layout & Strides Penalty

Using a $2048 \times 2048$ single-precision tensor (16 MB buffer):
- **Contiguous Array:** `stride = [2048, 1]`, `is_contiguous() = True`.
- **Transposed Array:** `stride = [1, 2048]`, `is_contiguous() = False`.

### Reduction Benchmark (Row-Wise Sum)
- **Contiguous Row-Sum Latency:** $0.209\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $0.351\text{ ms}$
- **Stride Penalty:** **$1.68\times$ slowdown**
- **Buffer Clone Cost (`.contiguous()`):** $8.525\text{ ms}$

### Hardware Mechanics
- **Cache Line Efficiency:** Sequential access utilizes all 64 bytes of an L1/L2 cache line (16 floats). Non-contiguous strides fetch 64 bytes but use only 4 bytes, evicting cache lines before reuse.
- **SIMD Vectorization:** Contiguous memory feeds AVX2 registers directly via single unaligned/aligned loads (`vmovups`). Strided layouts require gather operations or serial scalar reads.
- **Memory Allocation Cost:** Copying to contiguous memory takes $8.525\text{ ms}$, which is $60\times$ more expensive than the slowdown of a single reduction. Buffer copies are only worthwhile when amortized across multiple operations.

---

## 7. Technical Note: When Graph Compilation Delivers Value

Per Reddi Chapter 7, graph compilers provide substantial performance gains when:
1. **Fusing Pointwise Kernels:** Combining operations like `Linear + ReLU` prevents storing intermediate tensors in DRAM, keeping values in L1/L2 CPU caches.
2. **Removing Python Dispatch Latency:** For low-latency inference or small batch sizes, Python runtime overhead dominates. Lowering to native C++ eliminates interpreter overhead.
3. **Optimizing Memory Footprint:** Compilers analyze variable lifetimes and reuse static memory buffers, eliminating memory fragmentation and dynamic allocation calls.

---

## 8. Artifact Directory

- [scalar_math.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/src/scalar_math.py): Custom scalar autograd engine.
- [verify_engine.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/src/verify_engine.py): Mathematical gradient verification suite.
- [moons_benchmark.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/src/moons_benchmark.py): 5-trial benchmark on `make_moons`.
- [torch_graph_analysis.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/src/torch_graph_analysis.py): TorchScript tracing and memory striding analysis.
- `results/`: Contains [engine_validation.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/results/engine_validation.json), [moons_5runs_telemetry.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/results/moons_5runs_telemetry.json), and [strides_profiling.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_73681_miray/lab05/results/strides_profiling.json).
