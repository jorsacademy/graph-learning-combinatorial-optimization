# Contributing

Contributions should preserve the distinction between generative-policy learning, mathematical feasibility, exact reference computation, and empirical evaluation.

## Required checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Domain or environment changes

Changes to the state graph, action set, stop transition, or backward policy require tests for:

- reachability of every supported terminal object;
- constructive feasibility;
- acyclicity and finite horizon;
- trajectory multiplicity;
- exact state-flow versus terminal-partition agreement.

## Reward changes

A reward must remain strictly positive on every supported terminal object. Document its scale, numerical range, and effect on the target distribution. Do not substitute a top-one optimization objective for a reward-proportional target without changing the project claims.

## Model changes

New encoders should retain permutation tests. New balance objectives must be implemented alongside, not silently in place of, trajectory balance. Report parameter counts and keep exact oracle paths independent of neural code.

## Benchmark changes

Do not tune against frozen test seeds. New metrics must state whether they measure distribution fidelity, solution quality, diversity, reliability, or runtime. Avoid one weighted score that hides trade-offs.

## Checkpoints and data

Do not add pickle-based model loading. Corpus fingerprints must exclude runtime values and include all mathematical instance content.
