# Graph Attention Network on Cora

A compact Graph Attention Network (GAT) implementation for node classification on the Cora citation network using PyTorch and PyTorch Geometric.

## Overview

This project trains a two-layer GAT on the Cora dataset. The first layer uses multiple attention heads and the output layer produces class logits for node classification.

The implementation uses sparse graph connectivity through `edge_index` and `torch_geometric.nn.GATConv`. This avoids constructing a dense `N x N` attention tensor for every attention head, which is substantially more memory-efficient for citation graphs.

## Features

- Multi-head Graph Attention Network
- Cora dataset via `torch_geometric.datasets.Planetoid`
- Automatic CPU/CUDA device selection
- Training and validation metrics
- Test-set evaluation
- First-layer attention visualization around a selected node
- Bounded graph visualization to avoid excessively expensive plotting
- Command-line hyperparameters

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your system needs a platform-specific PyTorch build, install PyTorch using the instructions for your CUDA/CPU environment first, then install the remaining requirements.

## Usage

Run the default experiment:

```bash
python main.py
```

Run without plots:

```bash
python main.py --no-plots
```

Change selected hyperparameters:

```bash
python main.py --epochs 300 --hidden 8 --heads 8 --dropout 0.6 --lr 0.005
```

Visualize attention around another Cora node:

```bash
python main.py --node 42
```

## Default configuration

| Parameter | Value |
|---|---:|
| Dataset | Cora |
| Hidden channels per head | 8 |
| Attention heads | 8 |
| Dropout | 0.6 |
| Learning rate | 0.005 |
| Weight decay | 5e-4 |
| Epochs | 200 |

## Model

The architecture follows the standard two-layer GAT pattern:

1. Multi-head GAT layer
2. ELU activation
3. Dropout
4. Single-head output GAT layer
5. Log-softmax for node classification

The model is trained with negative log-likelihood loss on the Cora training mask and evaluated on the validation and test masks supplied by the Planetoid dataset.

## Why this version is memory-efficient

A naive GAT implementation may explicitly construct pairwise node representations for all node pairs, causing quadratic memory usage with respect to the number of nodes. This implementation instead computes attention only over graph edges through PyTorch Geometric's sparse `edge_index` representation.

## Files

- `main.py` — model, training, evaluation, and visualization
- `requirements.txt` — Python dependencies

## Reference

Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR 2018.
