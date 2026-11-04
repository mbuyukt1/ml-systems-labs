# Machine Learning Systems Engineering — Coursework Portfolio

**Student:** Miray Büyüktepe  
**Student ID:** 73681  
**Curriculum Textbook:** *Vijay Janapa Reddi, Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems.*  
**Environment Specification:** Python 3.11.8 (Pinned Stack)

---

## Portfolio Organization

This repository implements five laboratory assignments following the standardized structural specification:

    ml-systems-labs/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── lab01/  # Environment and System Measurements
    ├── lab02/  # Neural Network from First Principles
    ├── lab03/  # Architectures and Their Costs
    ├── lab04/  # Data Pipeline and Quality Assurance
    └── lab5/   # Frameworks and Automatic Differentiation
        ├── README.md
        ├── report.md
        ├── src/
        └── results/

Each lab contains its report, source code, and supporting results.

---

## Laboratory Modules

| Module | Title | Reddi Textbook | Delivered Evidence | Status |
| :---: | :--- | :--- | :--- | :---: |
| Lab 01 | Environment and First System Measurements | Ch. 1, 2, 12 | Pinned versions, latency distributions, RSS memory, edge/TinyML budget matrix | Verified |
| Lab 02 | A Neural Network from Scratch | Ch. 3 | Matrix calculus chain rule, 10/10 gradient verification, GEMM profiling | Verified |
| Lab 03 | Architectures and Their Costs | Ch. 4 | PyTorch MLP, CNN, Attention; hand FLOPs vs thop, Pareto trade-off curves | Verified |
| Lab 04 | Data Pipeline and Quality Engineering | Ch. 5, 6 | OpenML Adult dataset, zero-leakage preprocessing, feature synthesis, duplicate audit | Verified |
| Lab 5 | Frameworks and Automatic Differentiation | Ch. 7 | Scalar reverse-mode engine, gradient references, make-moons training timing, eager/traced and tensor-layout profiling | Complete |

---

## Execution Guide

    git clone https://github.com/mbuyukt1/ml-systems-labs.git
    cd ml-systems-labs
    python -m venv .venv
    # Activate .venv, then:
    pip install -r requirements.txt
    python lab01/src/baseline_experiments.py
    python lab02/src/experiment_runner.py
    python lab03/src/evaluate_efficiency.py
    python lab04/src/pipeline_runner.py
    python lab5/src/run_experiments.py
