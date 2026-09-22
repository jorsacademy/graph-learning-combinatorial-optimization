# Architecture

## Separation of responsibilities

The implementation separates five obligations:

1. **Domain validity:** graph and candidate decisions are typed and audited.
2. **Constructive feasibility:** the MDP masks every conflicting add action.
3. **Generative learning:** a graph policy and log-partition head are trained with trajectory balance.
4. **Independent reference computation:** complete enumeration and an exact state-flow recursion define the target law.
5. **Evaluation:** terminal distribution, objective quality, diversity, residuals, and runtime are reported independently.

The neural network is not used to certify feasibility or to define the benchmark target.

## State graph

A state is `(selected_mask, stopped)`.

- The initial state is `(0, False)`.
- A vertex action adds one currently available vertex.
- `STOP` maps `(S, False)` to `(S, True)`.
- Terminal states have no outgoing actions.

Every add action increases the selected-set cardinality, so the nonterminal state graph is acyclic. The maximum trajectory length is `|V| + 1`, including `STOP`.

## Backward policy

For a child with `k` selected vertices, every one of its `k` vertices can be removed while preserving independence. The declared backward probability is `1 / k`. The backward probability of the terminal `STOP` edge is one.

The backward policy is fixed rather than learned. This makes trajectory accounting directly inspectable and permits an independent exact flow recursion.

## Features

Per-node features:

- vertex weight divided by total graph weight;
- vertex weight divided by maximum vertex weight;
- normalized degree;
- inverse degree;
- selected indicator;
- available indicator;
- blocked indicator;
- bias.

Global features:

- normalized log vertex count;
- edge density;
- weight coefficient of variation;
- selected fraction;
- available fraction;
- scaled reward temperature.

## Encoder

Node features pass through a shared MLP. Each message-passing round computes mean neighbor embeddings from a row-normalized adjacency matrix, concatenates the current node embedding, neighbor embedding, and global graph context, then applies a shared update and layer normalization.

Graph context is formed from mean node pooling, max node pooling, and the encoded global features.

## Heads

- A shared node head returns one forward-action logit per vertex.
- A graph head returns the `STOP` logit.
- A second graph head predicts `log Z(G, beta)` from the initial state.

Invalid vertex logits are replaced by the minimum representable floating-point value before the categorical distribution is constructed.

## Training algorithms

### Trajectory balance

The model minimizes the mean squared TB residual over sampled trajectories. Training trajectories are drawn from an epsilon mixture of the model policy and a uniform law over valid actions. The residual itself always uses model-policy log probabilities.

### REINFORCE control

A separate copy of the same architecture maximizes expected terminal log reward using a moving baseline and entropy regularization. Its `log Z` head is not used. This control is intentionally mode-seeking and is evaluated with the same distributional metrics.

## Safe serialization

Neural weights use `safetensors`. Metadata records:

- checkpoint schema version;
- feature schema version;
- model configuration;
- algorithm;
- training and validation corpus fingerprints;
- training summary.

Checkpoint loading rejects missing metadata, incompatible schema versions, malformed model configurations, or tensor-shape mismatches.
