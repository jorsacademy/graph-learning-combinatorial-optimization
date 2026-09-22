# GraphSAGE Cora Node Classification

A compact educational implementation of **GraphSAGE** for semi-supervised node classification on the **Cora citation network** using PyTorch Geometric.

The project trains a GraphSAGE model, selects the best checkpoint using validation accuracy, evaluates the selected model on the test split, visualizes hidden node embeddings with t-SNE, and includes a small Zachary's Karate Club example to illustrate neighborhood message passing.

## Features

- GraphSAGE node classification with `torch_geometric.nn.SAGEConv`
- Cora dataset loading with normalized node features
- CPU / CUDA device selection
- Reproducible random seeds
- Validation-based best-model checkpointing
- Final test evaluation after model selection
- Training-loss and validation-accuracy plots
- t-SNE visualization of learned hidden representations
- 1-hop and 2-hop neighborhood visualization on the Karate Club graph
- Short notes on common GraphSAGE aggregator variants

## Dataset

Cora is a citation-network benchmark in which:

- nodes represent scientific publications,
- edges represent citation links,
- node features are bag-of-words representations,
- labels correspond to publication topics.

PyTorch Geometric downloads the dataset automatically on the first run.

## Installation

```bash
git clone https://github.com/jorsacademy/graphsage-cora-node-classification.git
cd graphsage-cora-node-classification
pip install -r requirements.txt
```

Depending on your operating system, CUDA version, and PyTorch installation, you may prefer to install PyTorch and PyTorch Geometric using their official platform-specific instructions first.

## Run

```bash
python main.py
```

The script will:

1. download/load Cora,
2. print dataset statistics,
3. train GraphSAGE for 200 epochs,
4. retain the model state with the best validation accuracy,
5. report the final test accuracy of that selected model,
6. display training and embedding visualizations,
7. visualize GraphSAGE neighborhood propagation on the Karate Club graph.

## Model

The default architecture is a two-layer GraphSAGE network:

```text
Input node features
        |
        v
SAGEConv(input -> 64)
        |
      ReLU
        |
     Dropout
        |
        v
SAGEConv(64 -> classes)
        |
        v
Class logits
```

The optimizer is Adam with:

- learning rate: `0.01`
- weight decay: `5e-4`
- dropout: `0.5`
- epochs: `200`

## Evaluation protocol

The training loss is computed only on nodes in the training mask. Validation accuracy is used for checkpoint selection. The test split is evaluated after restoring the best validation checkpoint, avoiding the use of test accuracy as a model-selection criterion.

## Why GraphSAGE?

GraphSAGE learns neighborhood aggregation functions rather than a fixed embedding for every node. This makes it suitable for **inductive learning**, where learned aggregation functions can be applied to nodes or graphs not explicitly observed during training.

PyTorch Geometric's `SAGEConv` uses mean aggregation by default.

## Project structure

```text
graphsage-cora-node-classification/
├── main.py
├── requirements.txt
└── README.md
```

## Notes

The t-SNE visualization is intended for qualitative inspection only. Different environments, library versions, and hardware can produce small numerical differences even when random seeds are fixed.

## License

This repository is provided as an educational example. Add a license file if you intend to distribute or reuse it under a specific open-source license.
