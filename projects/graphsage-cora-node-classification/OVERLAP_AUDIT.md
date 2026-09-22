# Repository Overlap Audit — Foundational GNN Examples

This document records overlap without merging, archiving, renaming, or deleting repositories.

## `graphsage-cora-node-classification`

**Keep separate.**

Educational objective: GraphSAGE and inductive neighborhood aggregation.

Distinctive elements include validation-based checkpointing, hidden-embedding visualization, and neighborhood-propagation demonstrations.

## `graph-attention-network-cora`

**Keep separate.**

Educational objective: Graph Attention Networks and learned edge/neighborhood attention.

Distinctive elements include multi-head GAT layers and first-layer attention visualization.

## Audit decision

Although both repositories use Cora and PyTorch Geometric, they should not be consolidated. The repeated dataset is useful because it controls the benchmark while changing the architecture.

A single multi-architecture repository would reduce repo count but weaken the one-concept-per-project educational structure. The current separation is therefore intentional and justified.
