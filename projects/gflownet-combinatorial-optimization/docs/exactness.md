# Exactness and Reliability Contract

## What is exact

For instances at or below the configured enumeration limit:

- every independent set is enumerated by deterministic include/exclude recursion;
- every enumerated bit mask is checked for independence;
- the exact maximum-weight objective and all optimal masks are identified;
- the reward-proportional terminal law is normalized over the complete feasible support;
- the state-flow recursion computes an exact sequential GFlowNet policy for the declared backward law;
- the root state flow is checked against the terminal partition function;
- every generated sample is checked against the exact support;
- objective gaps are computed against the exact optimum.

“Exact” is subject to ordinary floating-point evaluation of positive weights, exponentials, logarithms, and normalized probabilities.

## Constructive feasibility

At state `S`, a vertex is available only when it is not selected and has no selected neighbor. The environment refuses every other add action. Therefore every reachable nonterminal state is independent.

The model mask is not treated as the sole proof. Every terminal mask is independently checked with the graph adjacency masks, and the benchmark aborts if a sampler emits an infeasible state.

## Terminal target law

The declared target is

\[
p^\star(S)=\frac{R(S)}{\sum_{T\in\mathcal I(G)}R(T)},
\]

where \(\mathcal I(G)\) is the complete independent-set family.

The benchmark never estimates this denominator from model samples on exact-test instances. It is computed from the complete support.

## Exact flow recursion

For the uniform remove-one-vertex backward policy,

\[
F(S)=R(S)+\sum_{v\in A(S)}\frac{F(S\cup\{v\})}{|S|+1}.
\]

The recursion is evaluated from larger sets to smaller sets through memoized depth-first search. Its root is compared with the terminal log-sum-exp partition. A mismatch is a hard error.

## What is not exact

- The learned forward policy is approximate.
- The learned `log Z` head is approximate.
- Empirical distributions use finite sample counts.
- A small sampled TB residual does not prove global terminal-law equality.
- No approximation ratio is provided for learned or heuristic samples.
- Exact support enumeration is not attempted beyond the configured vertex limit.

## Fail-closed checks

The system raises errors rather than silently accepting:

- non-finite graph weights, logits, losses, gradients, or probabilities;
- invalid graph edges or duplicate edges;
- malformed or tampered corpora;
- incompatible checkpoints;
- nonterminating trajectories;
- invalid forward actions;
- infeasible terminal masks;
- samples absent from the exact support;
- candidate objective values above the exact optimum beyond tolerance;
- disagreement between state-flow and terminal partitions.

## Interpretation of oracle controls

`target_distribution_oracle` samples directly from the exact terminal categorical law.

`exact_flow_policy_oracle` samples sequentially through the same add-or-stop MDP used by the neural model. Agreement between these two controls validates the trajectory multiplicity and backward-policy construction.

`uniform_exact_oracle` is not a target sampler unless `beta = 0`; it is a diversity reference.
