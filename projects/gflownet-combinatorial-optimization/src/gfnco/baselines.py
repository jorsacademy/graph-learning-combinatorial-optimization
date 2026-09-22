"""Non-neural and exact-oracle sampling controls."""

from __future__ import annotations

import math

import numpy as np

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import ConstructionState, IndependentSetEnvironment
from gfnco.oracle import ExactFlowPolicy, ExactSolutionSet, TargetDistribution


def greedy_weight_degree_mask(problem: WeightedGraphProblem) -> int:
    selected = 0
    while True:
        available = problem.available_mask(selected)
        if available == 0:
            return selected
        candidates = [vertex for vertex in range(problem.vertex_count) if available & (1 << vertex)]
        vertex = max(
            candidates,
            key=lambda index: (
                problem.weights[index] / (1.0 + problem.degrees[index]),
                problem.weights[index],
                -index,
            ),
        )
        selected = problem.add_vertex(selected, vertex)


def repeated_greedy_samples(
    problem: WeightedGraphProblem,
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    mask = greedy_weight_degree_mask(problem)
    return tuple(mask for _ in range(sample_count))


def random_sequential_samples(
    problem: WeightedGraphProblem,
    *,
    sample_count: int,
    seed: int,
    stop_probability: float = 0.25,
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < stop_probability <= 1.0:
        raise ValueError("stop_probability must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    environment = IndependentSetEnvironment(problem)
    samples: list[int] = []
    for _ in range(sample_count):
        state = environment.initial_state
        for _ in range(problem.vertex_count + 1):
            available = [
                action
                for action in environment.valid_actions(state)
                if action != environment.stop_action
            ]
            stop = not available or rng.random() < stop_probability
            action = (
                environment.stop_action
                if stop
                else int(available[int(rng.integers(0, len(available)))])
            )
            state = environment.transition(state, action).current
            if state.stopped:
                samples.append(environment.terminal_mask(state))
                break
        else:  # pragma: no cover - protected by the acyclic horizon
            raise RuntimeError("random baseline failed to terminate")
    return tuple(samples)


def reward_biased_sequential_samples(
    problem: WeightedGraphProblem,
    beta: float,
    *,
    sample_count: int,
    seed: int,
    stop_logit: float = 0.0,
) -> tuple[int, ...]:
    """A local softmax heuristic that does not correct for trajectory multiplicity."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    if not math.isfinite(stop_logit):
        raise ValueError("stop_logit must be finite")
    rng = np.random.default_rng(seed)
    environment = IndependentSetEnvironment(problem)
    samples: list[int] = []
    for _ in range(sample_count):
        state = ConstructionState()
        for _ in range(problem.vertex_count + 1):
            actions = environment.valid_actions(state)
            logits = np.asarray(
                [
                    stop_logit
                    if action == environment.stop_action
                    else beta * problem.weights[action] / problem.total_weight
                    for action in actions
                ],
                dtype=float,
            )
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(np.sum(probabilities))
            action = actions[int(rng.choice(len(actions), p=probabilities))]
            state = environment.transition(state, action).current
            if state.stopped:
                samples.append(environment.terminal_mask(state))
                break
        else:  # pragma: no cover - protected by the acyclic horizon
            raise RuntimeError("reward-biased baseline failed to terminate")
    return tuple(samples)


def uniform_exact_samples(
    exact: ExactSolutionSet,
    *,
    sample_count: int,
    seed: int,
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(exact.masks), size=sample_count)
    return tuple(exact.masks[int(index)] for index in indices.tolist())


def target_oracle_samples(
    target: TargetDistribution,
    *,
    sample_count: int,
    seed: int,
) -> tuple[int, ...]:
    return target.sample(np.random.default_rng(seed), sample_count)


def exact_flow_policy_samples(
    policy: ExactFlowPolicy,
    *,
    sample_count: int,
    seed: int,
) -> tuple[int, ...]:
    return policy.sample(np.random.default_rng(seed), sample_count)
