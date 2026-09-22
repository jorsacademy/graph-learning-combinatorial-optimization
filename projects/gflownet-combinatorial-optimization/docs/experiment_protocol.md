# Experiment Protocol

## Data separation

Training, validation, and every evaluation scenario use disjoint deterministic seed ranges. A complete graph instance belongs to one split only. There is no vertex-level or trajectory-level split of the same graph across training and evaluation.

## Training distribution

The default protocol uses moderate-density Erdős–Rényi graphs, uniformly distributed positive weights, and a range of reward temperatures. The GFlowNet and REINFORCE control receive:

- the same graph corpus;
- the same number of optimization steps;
- the same batch size;
- the same architecture class;
- disjoint but deterministic random seeds.

## Evaluation scenarios

The frozen protocol reports:

1. `in_distribution`;
2. `size_shift`;
3. `sparse_graph`;
4. `dense_graph`;
5. `clustered_graph`;
6. `weight_shift`;
7. `combined_shift`;
8. `beta_shift`.

Each scenario has a separate corpus fingerprint. Results are aggregated by `scenario|method`, not only by method.

## Sampling budget

Every method receives the same number of terminal samples per graph. Deterministic greedy output is repeated to the same budget so that empirical-distribution metrics expose its mode collapse instead of receiving a special metric definition.

## Distribution metrics

Total variation and Jensen–Shannon divergence compare the empirical sample law with the exact reward-proportional terminal law over the full feasible support.

Target-mass coverage measures the exact target probability assigned to at least one observed sample. It distinguishes broad support coverage from repeatedly sampling low-probability states.

Probability log-correlation and slope use a documented finite-sample smoothing constant and are diagnostics, not proofs of calibration.

## Quality and diversity metrics

Quality metrics include mean and best objective, exact optimum hit rate, and objective-sense-specific optimality gap. Diversity metrics include unique rate, entropy, effective sample size, pairwise Hamming distance, and coverage of distinct solutions within 95% of the exact optimum.

No single weighted score combines these quantities. Distribution fidelity, optimization quality, and diversity remain separate.

## Oracle references

The direct target sampler and exact flow-policy sampler retain ordinary Monte Carlo error. Their observed total variation and Jensen–Shannon divergence provide sample-budget reference floors.

## Runtime

Sampler runtime is reported but should not be interpreted as a universal speed comparison. Exact enumeration cost is excluded from per-sampler runtime because it is a common evaluation oracle. Small CPU graphs may favor direct exact methods.

## Statistical reporting

The initial repository reports means across deterministic test instances. Publication-grade use should add:

- multiple training seeds;
- confidence intervals or bootstrap intervals;
- paired tests on per-instance metrics;
- scaling experiments beyond the exact-enumeration domain;
- external graph benchmarks;
- hyperparameter-selection separation from the final test set.
