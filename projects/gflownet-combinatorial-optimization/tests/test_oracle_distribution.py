from __future__ import annotations

import math

import numpy as np
import pytest

from gfnco.domain import WeightedGraphProblem
from gfnco.oracle import (
    build_exact_flow_policy,
    build_target_distribution,
    enumerate_independent_sets,
)


def test_exact_enumerator_on_path(path_problem: WeightedGraphProblem) -> None:
    exact = enumerate_independent_sets(path_problem)
    assert exact.masks == (0, 1, 2, 4, 5)
    assert exact.objectives == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert exact.optimum_objective == 4.0
    assert exact.optimum_masks == (5,)
    assert exact.objective_by_mask()[5] == 4.0


def test_target_distribution_is_exact_and_beta_conditional(
    path_problem: WeightedGraphProblem,
) -> None:
    exact = enumerate_independent_sets(path_problem)
    uniform = build_target_distribution(path_problem, exact, beta=0.0)
    assert np.allclose(uniform.probabilities, np.full(5, 0.2))
    target = build_target_distribution(path_problem, exact, beta=8.0)
    assert math.isclose(sum(target.probabilities), 1.0)
    assert target.probabilities[-1] == max(target.probabilities)
    samples = target.sample(np.random.default_rng(5), 20)
    assert len(samples) == 20
    assert set(samples) <= set(exact.masks)


def test_oracle_rejects_mismatched_problem(path_problem: WeightedGraphProblem) -> None:
    exact = enumerate_independent_sets(path_problem)
    other = WeightedGraphProblem("other", (1.0, 2.0, 4.0), ((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="different problem"):
        build_target_distribution(other, exact, beta=1.0)
    large = WeightedGraphProblem("large", tuple(1.0 for _ in range(25)), ())
    with pytest.raises(ValueError, match="limited"):
        enumerate_independent_sets(large)


def test_exact_state_flow_matches_terminal_partition(path_problem: WeightedGraphProblem) -> None:
    exact = enumerate_independent_sets(path_problem)
    target = build_target_distribution(path_problem, exact, beta=5.0)
    policy = build_exact_flow_policy(path_problem, exact, beta=5.0)
    assert math.isclose(policy.log_partition, target.log_partition, abs_tol=1e-9)
    root = policy.forward_probabilities(0)
    assert math.isclose(sum(root.values()), 1.0)
    samples = policy.sample(np.random.default_rng(9), 200)
    assert all(path_problem.is_independent(mask) for mask in samples)
