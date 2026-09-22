# Graph Learning and GNN-for-Optimization Research Series

This file maps graph-learning repositories from foundational node-classification examples to solver-guidance and neural combinatorial optimization. It is an index only: each repository remains independent.

## Foundational GNN architectures

- `graphsage-cora-node-classification` — GraphSAGE on Cora with inductive neighborhood aggregation, checkpoint selection, embedding visualization, and neighborhood demonstrations.
- `graph-attention-network-cora` — GAT on Cora with multi-head attention and attention visualization.

These two should remain separate because the educational objective is architecture-specific: neighborhood aggregation versus learned attention weights.

## GNN guides for Industrial Engineering and OR

- `gnn-for-industrial-engineering-and-or-optimization` — English educational guide to graph representations and GNN roles in OR/optimization.
- `gnn-ile-yoneylem-arastirmasi-ve-optimizasyon` — Turkish educational guide covering the same broad subject in a separate language-specific learning repository.

Both language repositories are intentionally preserved as independent educational resources.

## Direct graph-based combinatorial optimization

- `gnn-minimum-vertex-cover` — GNN applied directly to a graph combinatorial problem.
- `graph-neural-solver-combinatorial-optimization` — broader graph-neural solver setting for combinatorial optimization.
- `neural-algorithmic-reasoning-combinatorial-optimization` — algorithmic reasoning over combinatorial structures.

## GNN guidance inside exact optimization

- `learning-to-branch-mip-gnn-scip-pytorch` — variable-constraint graph representation for learned branching inside SCIP.
- `learning-to-branch-milp` — includes a bipartite GNN branching policy in a transparent branch-and-bound sandbox.
- `gnn-guided-generalized-assignment-variable-fixing-pytorch` — learned variable fixing/primal guidance.
- `learning-to-cut-milp` — learned cut selection in the MIP-solver family.
- `neural-diving-mip-solution-prediction` — learned primal/variable guidance.

## Routing/NCO bridge

- `neural-combinatorial-optimization-tsp` — attention-based neural combinatorial optimization benchmark.
- `capacitated-vrp-rl4co-pomo-attention-model-python` — routing with modern attention/POMO tooling.
- `diffusion-neural-combinatorial-optimization-tsp-pytorch` — diffusion-based combinatorial solution generation.

## Portfolio rule

The portfolio should preserve the progression:

`GNN architecture fundamentals -> graph representation for OR -> direct neural CO -> GNN-guided exact solvers`.

Repositories should not be merged merely because they all use message passing or graph data. The role of the GNN inside the decision pipeline is the more important distinction.
