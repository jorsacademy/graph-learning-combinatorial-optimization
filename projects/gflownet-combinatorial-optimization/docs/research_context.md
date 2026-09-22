# Research Context

GFlowNets learn a stochastic constructive policy whose terminal probability is proportional to a positive reward. This differs from conventional reinforcement learning, which normally emphasizes expected reward and may collapse toward a small number of modes.

The repository follows four research lines:

1. **Generative Flow Networks.** Bengio et al. introduced flow-network objectives for diverse candidate generation over compositional spaces.
2. **Trajectory balance.** Malkin et al. introduced a trajectory-level conservation equation with improved long-horizon credit assignment.
3. **GFlowNets for graph combinatorial optimization.** Zhang et al. developed conditional GFlowNet formulations for graph CO tasks and emphasized diverse high-quality candidates.
4. **GFlowNet-guided search.** GFACS combines a learned GFlowNet prior with ant-colony-style stochastic search.

This implementation does not reproduce those paper-scale architectures or results. It contributes a small audit-oriented laboratory in which the complete target distribution is available.

## Why maximum-weight independent set?

MWIS supplies:

- an NP-hard graph optimization problem;
- a simple constructive feasibility mask;
- many trajectories leading to the same terminal subset;
- variable graph sizes and structural shifts;
- exact support enumeration on small graphs;
- a clear quality–diversity trade-off.

The multi-trajectory property is important. A direct autoregressive model can accidentally learn trajectory frequency rather than terminal reward unless path multiplicity is treated correctly.

## Why include an exact flow policy?

Most neural CO repositories compare objective values with an exact optimum. That is insufficient for a generative sampler whose primary claim concerns a distribution. The exact flow recursion provides a sequential reference with the same MDP and backward policy as the learned model. It can expose mistakes in stop-edge treatment or reverse-transition probabilities.

## Related primary references

- Bengio et al., *Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation*, 2021: https://arxiv.org/abs/2106.04399
- Malkin et al., *Trajectory Balance: Improved Credit Assignment in GFlowNets*, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract-Conference.html
- Zhang et al., *Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets*, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract.html
- Kim et al., *Ant Colony Sampling with GFlowNets for Combinatorial Optimization*, 2024: https://arxiv.org/abs/2403.07041

## Natural extensions

- subtrajectory balance and detailed-balance ablations;
- learned or pessimistic backward policies;
- replay-buffer and off-policy prioritization studies;
- local-search or ant-colony posterior refinement;
- graph-transformer encoders;
- conditional multi-problem training;
- external MWIS benchmarks;
- approximate counting and partition estimation at larger scales;
- multi-objective reward conditioning.
