# Model Card

## Intended use

Research and teaching on reward-proportional generative sampling for small weighted graph optimization problems.

## Model

A compact graph neural policy with shared mean-neighbor message passing, global pooling, vertex/stop action heads, and an instance-conditioned log-partition head.

## Training objective

Trajectory balance for the primary model. A separate REINFORCE model is provided as a mode-seeking control.

## Inputs

- positive weighted undirected graph;
- current feasible selected mask;
- nonnegative reward temperature.

## Outputs

A categorical distribution over feasible vertex additions and `STOP`. Repeated sampling yields feasible independent sets.

## Guarantees

- invalid add actions are masked;
- every terminal sample is independently checked for feasibility;
- exact target metrics are available only when complete support enumeration succeeds.

The neural policy has no optimality or distributional guarantee.

## Known limitations

- synthetic graph training;
- small exact-test graphs;
- no local search;
- no external benchmark tuning;
- no calibrated uncertainty estimate;
- no global verification of the neural terminal law;
- potential mode collapse or poor transfer under distribution shift.

## Licensing

Source-available for non-commercial use under PolyForm Noncommercial 1.0.0.
