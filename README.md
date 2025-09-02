# ROSC-S 🧠
**Robust multi-modal Alzheimer’s diagnosis using Schatten-p norm and graph learning**

> 📍 IEEE 2025 Submission  
> 📎 [Paper] | 📂 [Open-source Code]

---

## 🚀 Overview

ROSC-S is a robust, graph-regularized multi-modal learning framework designed for Alzheimer's disease diagnosis. It unifies representation learning, similarity matrix fusion, and feature disentanglement via Schatten-p norm regularization.

---

## 🧩 Key Features
- Low-rank tensor graph construction
- Orthogonal soft constraint projection
- Multi-modal fusion via adaptive Laplacian learning
- Fully modular + research-ready implementation

---

## 📦 Project Structure

```bash
ROSC-S/
├── train.py                # Main training script
├── config.py               # Hyperparameter configs
├── modules/                # Core algorithm modules
│   ├── optimization.py
│   ├── update_steps.py
│   ├── data_preprocessing.py
│   ├── initialization.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── utils.py
├── data/                   # Loaders or demos
├── requirements.txt
├── LICENSE

