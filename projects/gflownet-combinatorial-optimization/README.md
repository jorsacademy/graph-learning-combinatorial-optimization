# GFlowNet Combinatorial Optimization

[![CI](https://github.com/jorsacademy/gflownet-combinatorial-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/gflownet-combinatorial-optimization/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of a **conditional trajectory-balance GFlowNet** for sampling diverse, high-quality solutions to the maximum-weight independent-set problem.

The repository is organized around a stricter claim than “the neural sampler found a good solution”:

> On graph sizes where the feasible support can be completely enumerated, compare the learned terminal law directly with the exact reward-proportional target distribution while independently auditing feasibility, solution quality, diversity, trajectory-balance residuals, and distribution shift.

The implementation includes an exact state-flow GFlowNet oracle, a mode-seeking REINFORCE control, non-neural sampling baselines, complete independent-set enumeration, safe checkpoints, deterministic corpora, and an end-to-end research protocol.

## Research question

For a weighted graph \(G=(V,E)\), can an instance-conditioned GFlowNet learn to sample independent sets according to

\[
p^\star(S\mid G,\beta)
=\frac{R_\beta(S)}{Z(G,\beta)},
\qquad
R_\beta(S)
=\exp\left(
\beta\frac{w(S)}{\sum_{i\in V}w_i}
\right),
\]

while preserving hard combinatorial feasibility and maintaining useful diversity under changes in graph size, density, topology, weight distribution, and reward temperature?

The reward temperature \(\beta\) controls the quality–diversity trade-off:

- \(\beta=0\) gives the uniform law over feasible independent sets;
- larger \(\beta\) concentrates probability on high-weight sets;
- finite \(\beta\) still assigns positive probability to every feasible terminal object.

Because

\[
0\leq
\beta\frac{w(S)}{\sum_i w_i}
\leq\beta,
\]

the log reward is bounded and numerically stable.

## Claims boundary

This repository is a compact and auditable methodology benchmark. It does **not** claim:

- a frontier-scale generative foundation model;
- state-of-the-art maximum-weight independent-set performance;
- superiority to exact optimization on small graphs;
- an approximation guarantee for neural samples;
- exact reward-proportional sampling outside the enumerated test domain;
- reproduction of *Let the Flows Tell*, GFACS, or later hybrid-balance systems;
- universal support for arbitrary combinatorial optimization problems.

The core scientific question is whether the learned terminal distribution approaches the declared target law—not whether a stochastic model occasionally returns the optimum.

## Optimization problem

The benchmark uses maximum-weight independent set:

\[
\max_{x\in\{0,1\}^{|V|}}
\quad \sum_{i\in V} w_i x_i
\]

subject to

\[
x_i+x_j\leq 1
\qquad \forall (i,j)\in E.
\]

All weights are finite and strictly positive. A terminal bit mask is independently audited against every graph edge before it is scored.

## Constructive acyclic MDP

A state is a feasible selected-vertex mask. From state \(S\), the forward action set contains:

1. every vertex not selected and not adjacent to a selected vertex;
2. a dedicated `STOP` action.

Adding a vertex preserves independence by construction. `STOP` turns the current feasible subset into a terminal object. Every independent set is reachable through every permutation of its selected vertices followed by `STOP`.

```text
empty set
   │
   ├── add any available vertex
   │       │
   │       ├── add another non-conflicting vertex
   │       └── STOP
   │
   └── STOP  → empty terminal set
```

No feasibility repair is needed for model trajectories: invalid add actions are masked before sampling, and every returned terminal mask is checked again by the domain auditor.

## Backward policy and trajectory multiplicity

A terminal set of size \(k\) can be reached through \(k!\) vertex orderings. Ignoring this multiplicity produces the wrong terminal law.

For an add transition into a state containing \(k\) vertices, the declared backward policy is uniform over the \(k\) removable vertices:

\[
P_B(S_{t-1}\mid S_t)=\frac{1}{k}.
\]

The terminal `STOP` edge has backward probability one. The implementation records every backward log probability explicitly.

## Trajectory-balance objective

For a complete trajectory \(\tau=(s_0,\ldots,s_T=x)\), trajectory balance requires

\[
\log Z(G,\beta)
+
\sum_{t=0}^{T-1}\log P_F(s_{t+1}\mid s_t,G,\beta)
-
\sum_{t=0}^{T-1}\log P_B(s_t\mid s_{t+1})
-
\log R_\beta(x)
=0.
\]

The learned loss is the squared residual:

\[
\mathcal L_{\mathrm{TB}}(\tau)
=
\left[
\log Z
+
\sum_t\log P_F
-
\sum_t\log P_B
-
\log R
\right]^2.
\]

A small epsilon-mixture behavior policy broadens trajectory coverage during training. The TB residual is still evaluated with the learned forward-policy probabilities, which permits off-policy trajectory sampling without redefining the target equation.

## Exact state-flow oracle

Small graphs admit a second, independent oracle beyond direct terminal enumeration. Under the declared uniform backward policy, the exact state flow obeys

\[
F(S)
=
R_\beta(S)
+
\sum_{v\in A(S)}
\frac{F(S\cup\{v\})}{|S|+1},
\]

where \(A(S)\) is the valid add-action set. The first term is the terminal `STOP` flow. The second term follows from the backward probability at each child state.

This recursion yields exact forward probabilities:

\[
P_F(\mathrm{STOP}\mid S)=\frac{R_\beta(S)}{F(S)},
\]

\[
P_F(v\mid S)
=
\frac{F(S\cup\{v\})/(|S|+1)}{F(S)}.
\]

At the root,

\[
\log F(\varnothing)=\log Z(G,\beta).
\]

The code checks that the root state-flow partition equals the partition computed by complete terminal enumeration. The sequential exact-flow sampler is included as an oracle control; it tests the MDP and backward-policy accounting separately from a direct categorical sample over enumerated terminal sets.

## Neural architecture

The forward policy is an instance-conditioned graph network.

```text
static graph + current selected mask + beta
                  │
                  ▼
per-vertex state features
                  │
                  ▼
shared node encoder
                  │
                  ▼
mean-neighbor graph message passing
                  │
                  ▼
global mean/max pooling + graph context
                  │
          ┌───────┴────────┐
          ▼                ▼
vertex action logits    STOP logit
          │
          ▼
valid-action mask and categorical policy

initial-state graph context
          │
          ▼
learned log Z(G, beta)
```

Node features include normalized weight, normalized degree, inverse degree, selected/available/blocked indicators, and a bias term. Global features include graph size, edge density, weight dispersion, selected fraction, available fraction, and scaled \(\beta\).

The architecture is permutation equivariant with respect to vertex reindexing: node operations are shared, neighbor aggregation is symmetric, and graph pooling is permutation invariant. Regression tests permute vertices and verify that action logits permute accordingly while the `STOP` logit remains unchanged.

## Compared methods

The benchmark evaluates the following methods on identical graph instances and sample budgets:

| Method | Neural | Feasible by construction | Targets reward-proportional law | Uses exact support |
| --- | ---: | ---: | ---: | ---: |
| `gflownet_tb` | Yes | Yes | Yes, learned | No |
| `reinforce` | Yes | Yes | No; maximizes expected reward | No |
| `reward_biased_sequential` | No | Yes | Local softmax only | No |
| `random_sequential` | No | Yes | No | No |
| `greedy_weight_degree` | No | Yes | No; deterministic mode | No |
| `uniform_exact_oracle` | No | Yes | Only when \(\beta=0\) | Yes |
| `exact_flow_policy_oracle` | No | Yes | Yes, exact sequential law | Yes |
| `target_distribution_oracle` | No | Yes | Yes, direct terminal law | Yes |

The REINFORCE policy uses the same graph architecture and training-instance budget as the GFlowNet control. It optimizes expected terminal log reward with an entropy bonus and a moving baseline. This separates reward maximization from distribution matching: a high-quality but mode-collapsed policy can score well on objective value while performing poorly on total variation, target-mass coverage, and mode coverage.

## Exact-distribution evaluation

For every reported test graph, all feasible independent sets are enumerated. The exact target law is then available in closed form. The benchmark reports:

### Distribution fidelity

- total variation distance;
- Jensen–Shannon divergence;
- target probability mass covered by sampled support;
- sampled support coverage;
- correlation and slope between target and empirical log probabilities;
- exact versus learned \(\log Z\);
- mean absolute and root-mean-square TB residual.

### Solution quality

- mean and best independent-set weight;
- mean and best optimality gap;
- exact optimum hit rate;
- target expected objective;
- coverage of distinct solutions within 95% of the optimum.

### Diversity

- unique terminal count and rate;
- empirical entropy;
- effective sample size;
- normalized mean pairwise Hamming distance.

### Reliability and runtime

- independent feasibility rate;
- graph, reward-temperature, and support metadata;
- sampler runtime under the same sample budget.

The oracle samplers retain finite-sample error. They are not hard-coded to report zero total variation; they provide empirical reference floors for the chosen sample budget.

## Frozen distribution-shift protocol

The research command trains both neural policies on independent Erdős–Rényi graphs with:

- training vertex counts in a fixed interval;
- moderate edge density;
- uniformly distributed positive weights;
- a set of training reward temperatures.

It then evaluates disjoint seed ranges for:

1. in-distribution graphs;
2. larger graphs;
3. sparse graphs;
4. dense graphs;
5. clustered graphs;
6. lognormal weight shift;
7. simultaneous density and weight shift;
8. reward-temperature shift.

Scenarios are reported separately. A favorable average cannot hide failure on one distribution shift.

## Installation

```bash
python -m pip install -e ".[dev]"
```

CPU-only PyTorch is sufficient. The package does not require CUDA.

## CLI

### Generate a graph

```bash
gfnco generate \
  --vertices 12 \
  --edge-probability 0.30 \
  --regime in_distribution \
  --seed 42 \
  --output artifacts/problem.json
```

### Inspect exact support and target statistics

```bash
gfnco oracle \
  --input artifacts/problem.json \
  --beta 6 \
  --output artifacts/oracle.json
```

### Build deterministic corpora

```bash
gfnco collect \
  --instances 48 \
  --min-vertices 8 \
  --max-vertices 12 \
  --seed 1000 \
  --output artifacts/train.jsonl

gfnco collect \
  --instances 12 \
  --min-vertices 8 \
  --max-vertices 12 \
  --seed 2000 \
  --output artifacts/validation.jsonl
```

### Train trajectory balance

```bash
gfnco train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --algorithm trajectory_balance \
  --steps 1500 \
  --batch-size 8 \
  --betas 2 4 6 \
  --checkpoint artifacts/gflownet.safetensors \
  --output-report artifacts/gflownet-training.json
```

### Train the REINFORCE control

```bash
gfnco train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --algorithm reinforce \
  --steps 1500 \
  --batch-size 8 \
  --betas 2 4 6 \
  --checkpoint artifacts/reinforce.safetensors \
  --output-report artifacts/reinforce-training.json
```

### Sample solutions

```bash
gfnco sample \
  --input artifacts/problem.json \
  --checkpoint artifacts/gflownet.safetensors \
  --beta 6 \
  --samples 128 \
  --output artifacts/samples.json
```

### Compare samplers against the exact target

```bash
gfnco benchmark artifacts/test.jsonl \
  --gflownet-checkpoint artifacts/gflownet.safetensors \
  --reinforce-checkpoint artifacts/reinforce.safetensors \
  --beta 6 \
  --samples 512 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

### Run the complete protocol

```bash
gfnco research \
  --gflownet-checkpoint artifacts/research-gflownet.safetensors \
  --reinforce-checkpoint artifacts/research-reinforce.safetensors \
  --output-report artifacts/research-report.json
```

The frozen defaults are also recorded in [`configs/research_v1.json`](configs/research_v1.json).

## Reproducibility and checkpoint safety

- Graph generation and every sampling path use explicit seeds.
- Training and evaluation seed ranges are disjoint.
- Corpus fingerprints hash mathematical graph content, not runtime diagnostics.
- Corpus loading verifies record count and SHA-256 fingerprint.
- Checkpoints use `safetensors`; no pickle-based model deserialization is required.
- Checkpoints carry schema version, feature version, model configuration, algorithm, and corpus fingerprints.
- Non-finite logits, losses, gradients, target probabilities, or residuals fail closed.
- Tests run with deterministic CPU operations where supported.

## Repository layout

```text
src/gfnco/
├── domain.py       # weighted graph schema and feasibility audit
├── generator.py    # deterministic graph and weight shifts
├── dataset.py      # versioned corpora and fingerprints
├── environment.py  # add-or-stop constructive DAG
├── features.py     # graph/state feature construction
├── model.py        # graph policy, log-Z head, safe checkpoints
├── trajectory.py   # rollouts and TB accounting
├── training.py     # TB and REINFORCE training
├── oracle.py       # exact support, target law, exact state-flow policy
├── baselines.py    # non-neural and oracle controls
├── evaluation.py   # distribution, quality, diversity, and runtime metrics
├── experiment.py   # frozen train/shift protocol
└── cli.py          # command-line workflows
```

Additional methodological detail is available in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/exactness.md`](docs/exactness.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/research_context.md`](docs/research_context.md)
- [`docs/model_card.md`](docs/model_card.md)

## Tests and CI

GitHub Actions runs the following on Python 3.11 and 3.12:

```text
package installation and dependency check
Ruff lint
Ruff formatting check
strict mypy
branch-aware pytest coverage
collect → train TB → train REINFORCE → sample → exact benchmark smoke
```

Heavy research training is intentionally excluded from CI. CI uses tiny deterministic graphs while retaining the same code paths and exact-distribution checks.

## Methodological limitations

Complete support enumeration is exponential. Exact distribution metrics are therefore limited to small and medium synthetic graphs. The learned sampler can process larger graphs, but exact total variation and partition audits are unavailable there unless another certified counting method is supplied.

The model uses a compact mean-aggregation graph network rather than the architectures from larger GFlowNet CO systems. It does not use local search, ant-colony refinement, subtrajectory balance, learned backward policies, replay buffers, or distributed actors. These are appropriate follow-on studies, not hidden features of this version.

A low TB residual on sampled trajectories is necessary but not by itself sufficient evidence of global distributional accuracy. That is why the repository reports both trajectory residuals and exact terminal-distribution metrics.

## Research context

This implementation is positioned relative to:

- Bengio et al., [*Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation*](https://arxiv.org/abs/2106.04399), which introduced the GFlowNet framework for sampling compositional objects in proportion to reward;
- Malkin et al., [*Trajectory Balance: Improved Credit Assignment in GFlowNets*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract-Conference.html), which introduced the TB objective used here;
- Zhang et al., [*Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract.html), which developed conditional GFlowNets for graph CO;
- Kim et al., [*Ant Colony Sampling with GFlowNets for Combinatorial Optimization*](https://arxiv.org/abs/2403.07041), which combines a GFlowNet prior with parallel stochastic search.

The repository does not reproduce those systems. It isolates a smaller question: can reward-proportional terminal sampling be measured directly and audited end to end on an exactly enumerable graph-optimization domain?

## License

This repository is **source-available for non-commercial use** under the [PolyForm Noncommercial License 1.0.0](LICENSE). It is not described as OSI Open Source because commercial use is not granted.
