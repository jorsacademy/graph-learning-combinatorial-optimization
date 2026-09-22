# GNN Minimum Vertex Cover

A small PyTorch Geometric project that uses a Graph Convolutional Network (GCN) to learn node scores for the minimum vertex cover problem.

The neural network produces soft node-selection scores. Since thresholding probabilities alone does **not** guarantee a feasible vertex cover, the project includes a deterministic repair-and-prune post-processing step that guarantees every edge is covered and then removes redundant selected vertices when possible.

## Features

- Reproducible Erdős–Rényi graph generation
- Degree-based node features
- Vectorized edge-coverage loss
- Correct handling of graphs with no edges
- Guaranteed-feasible vertex-cover extraction via repair + prune
- NetworkX approximation baseline
- Exact branch-and-bound solver for small benchmark graphs
- GNN/OPT and NetworkX/OPT approximation-ratio benchmarks
- Pytest test suite
- GitHub Actions CI

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python vertex_cover_gnn.py
```

## Benchmark against the exact optimum

```bash
python benchmark.py
```

For each seeded graph, the benchmark reports:

- exact minimum vertex-cover size (`OPT`)
- repaired GNN cover size
- NetworkX approximation cover size
- `GNN / OPT` approximation ratio
- `NetworkX / OPT` approximation ratio

A ratio of `1.0` means the candidate matched the exact optimum. The included exact solver uses branch-and-bound and is intentionally capped at small graph sizes because minimum vertex cover is NP-hard.

## Test

```bash
python -m pytest -q
```

## Method

For node probabilities `p_u` and `p_v`, the soft probability that an edge `(u, v)` is covered is modeled as:

```text
p_u + p_v - p_u * p_v
```

Training balances a small expected cover size against penalties for insufficient edge coverage. After inference:

1. Nodes above the probability threshold are selected.
2. Any uncovered edge is repaired by selecting the endpoint with the larger model score.
3. Selected vertices are checked in ascending score order and removed whenever feasibility is preserved.

The final output is therefore always a valid vertex cover, although—as expected for an NP-hard problem—it is not guaranteed to be minimum.
