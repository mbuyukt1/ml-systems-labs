# Lab 5: Frameworks and Automatic Differentiation

**Student Researcher:** Miray Büyüktepe (Student ID: 73681)  
**Reading:** Reddi, Chapter 7, §§7.3, 7.4, 7.6-7.8

## 1. Goal

Build a scalar reverse-mode automatic differentiation engine, check its gradients against PyTorch and numerical estimates, and examine framework execution and tensor-layout overheads.

## 2. Method

1. Implemented Value nodes with scalar data, accumulated gradients, parent references, and a local derivative rule.
2. Implemented graph-building addition, multiplication, constant powers, ReLU, sigmoid, and logarithm operations.
3. Used a visited-set topological traversal, seeded the output gradient with 1, cleared stale gradients, and accumulated contributions in reverse order.
4. Compared the gradient of sum(relu((a*b+c)**2)) over eight values per input with PyTorch autograd and central finite differences (epsilon=1e-6).
5. Trained a 2→8→1 sigmoid classifier with binary cross entropy on 128 seeded two-moons points. Compared the custom engine with PyTorch SGD across five runs of 18 epochs.
6. Traced a small PyTorch model with torch.jit.trace, warmed each path, and measured five repeats of 300 calls. Timed matrix multiplication using a contiguous tensor and a same-shape strided view.

Run python lab5/src/run_experiments.py to reproduce the measurements. The exact environment and per-run values are recorded in results/lab5_results.json.

## 3. Results

### Gradient validation

| Comparison | Maximum absolute difference | Requirement |
| :--- | ---: | :---: |
| Custom engine vs PyTorch autograd | 0.0 | < 1e-6: Pass |
| Custom engine vs central finite differences | 1.61e-10 | < 1e-6: Pass |

### Training and execution

| Measurement | Result |
| :--- | ---: |
| Custom engine training median (5 runs) | 1.024 s |
| PyTorch training median (5 runs) | 1.611 s |
| Custom engine median training accuracy | 87.5% |
| PyTorch median training accuracy | 89.1% |
| Eager model median forward time | 113.71 µs |
| Traced model median forward time | 57.76 µs |
| Eager / traced time ratio | 1.97x |

### Tensor layout

The measured tensor had shape 256 x 256 and dtype float32. Its contiguous stride was (256, 1); the strided view had stride (512, 2).

| Matrix multiplication input | Median time |
| :--- | ---: |
| Contiguous | 259.29 µs |
| Non-contiguous view | 305.05 µs |
| View / contiguous ratio | 1.18x |

These timings are CPU measurements from the environment recorded in the JSON output. They describe this run and are not hardware-independent benchmarks.

## 4. Conclusions

1. Topological backpropagation with additive gradient updates matched both independent references within the required tolerance. Testing several elements together also exercises the sum node's contribution to the graph.
2. Tracing nearly halved the measured forward-call time for this small model after warm-up. Compilation or tracing benefits depend on workload size and runtime; this small CPU result should not be generalized to every model.
3. The strided view took about 18% longer for the measured matrix multiplication. Tensor layout can affect framework cost even when shape and dtype are unchanged.
4. The custom scalar engine trained this small model faster in this run, while PyTorch reached higher median accuracy. The custom engine favors transparency and graph-level learning; PyTorch's tensor operations and optimized kernels are better suited to larger workloads.

## 5. Rubric note

The supplied manual assigns weights of 25, 20, 25, 10, 15, and 20 points, which sum to 115. The report addresses all six listed requirements; confirm the intended total with the course staff if the point sum affects grading.
