# GraphSAGE Architecture Ablation

## Why this exists

The repository's primary `EdgeGNN` injects pairwise edge features into every
message-passing layer. That is a strong inductive bias for TSP because distances
are part of the combinatorial objective.

A vanilla GraphSAGE tutorial on Cora would not test that design choice. Instead,
this ablation adapts the GraphSAGE neighborhood-aggregation idea to the same TSP
edge-supervision and decoder pipeline.

The controlled question is:

> Does edge-aware message passing improve downstream TSP decisions relative to a
> node-only GraphSAGE encoder when training data, supervision, decoder, local
> search, model-selection protocol and evaluation instances are held fixed?

## Compared architectures

### `edge_gnn`

The repository's primary model. Pairwise edge features enter the learned message
function at every layer and the final edge scorer.

### `graphsage`

A GraphSAGE-style node encoder. For each city it:

1. builds a `k`-nearest-neighbor neighborhood from Euclidean distance;
2. averages neighboring node states;
3. combines self and neighborhood state through a learned update;
4. scores every candidate TSP edge from the two final node states plus pairwise
   distance.

Distance is therefore available for graph construction and final edge scoring,
but it is **not injected into the learned messages**. This isolates the main
architectural distinction without changing the downstream feasibility logic.

The implementation is pure PyTorch. `torch_geometric` is intentionally not added
as a dependency because the repository already has a small, transparent message-
passing stack and the ablation needs only mean aggregation.

## Experimental controls

Both neural models use:

- the same exact Held-Karp training labels;
- the same training, validation, test and OOD instance seeds;
- the same hidden dimension, depth, optimizer and epoch budget;
- the same validation-based checkpointing;
- the same independent model seeds;
- the same multi-start beam decoder;
- the same 2-opt post-refinement;
- the same exact optimality-gap computation.

Model seed selection is performed on validation **decision gap**, not test or OOD
performance.

The classical `nearest_neighbor + 2-opt` method remains in the output as a
non-neural reference.

## Run

```bash
pip install -e '.[dev,neural]'
python -m gnn_solver.architecture_ablation
```

The script reports, for `test`, `ood_10` and `ood_12`:

- selected model seed and validation gap;
- mean and median exact optimality gap;
- end-to-end neural-score + decoder + 2-opt latency;
- feasibility rate.

## Interpretation

A lower edge-prediction loss is not sufficient evidence that an architecture is
better. Promotion should be based on downstream optimality gap under the same
feasibility-preserving decoder, together with latency and OOD behavior.

If GraphSAGE matches `EdgeGNN`, the extra edge-aware message machinery may not be
justified on the current benchmark. If `EdgeGNN` wins consistently, the ablation
provides direct evidence that pairwise edge information belongs inside the
message-passing stack rather than only in the final scorer.
