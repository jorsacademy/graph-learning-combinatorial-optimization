from __future__ import annotations

import json

import numpy as np
import pytest

from gfnco.domain import WeightedGraphProblem, load_problem, save_problem
from gfnco.generator import GeneratorConfig, generate_problem, generate_problems


def test_domain_state_operations(path_problem: WeightedGraphProblem) -> None:
    assert path_problem.edges == ((0, 1), (1, 2))
    assert path_problem.adjacency_masks == (2, 5, 2)
    assert path_problem.degrees == (1, 2, 1)
    assert path_problem.available_mask(0) == 0b111
    first = path_problem.add_vertex(0, 0)
    assert first == 0b001
    assert path_problem.available_mask(first) == 0b100
    complete = path_problem.add_vertex(first, 2)
    assert complete == 0b101
    assert path_problem.is_independent(complete)
    assert path_problem.objective(complete) == 4.0
    assert path_problem.mask_to_decision(complete) == (1, 0, 1)
    assert path_problem.decision_to_mask([1.0, 0.0, 1.0]) == complete
    with pytest.raises(ValueError, match="not a valid"):
        path_problem.add_vertex(first, 1)


def test_decision_audit_and_validation(path_problem: WeightedGraphProblem) -> None:
    feasible = path_problem.audit_decision(np.asarray([1.0, 0.0, 1.0]))
    assert feasible.feasible
    assert feasible.conflicting_edges == ()
    conflict = path_problem.audit_decision(np.asarray([1.0, 1.0, 0.0]))
    assert not conflict.feasible
    assert conflict.conflicting_edges == ((0, 1),)
    fractional = path_problem.audit_decision(np.asarray([0.2, 0.0, 1.0]))
    assert not fractional.feasible
    with pytest.raises(ValueError, match="binary"):
        path_problem.decision_to_mask([0.2, 0.0, 1.0])
    with pytest.raises(ValueError, match="reward"):
        path_problem.log_reward(0b011, 3.0)
    with pytest.raises(ValueError, match="beta"):
        path_problem.log_reward(0, -1.0)


def test_problem_round_trip(tmp_path, path_problem: WeightedGraphProblem) -> None:
    destination = tmp_path / "problem.json"
    save_problem(path_problem, destination)
    loaded = load_problem(destination)
    assert loaded == path_problem
    assert loaded.fingerprint == path_problem.fingerprint
    payload = json.loads(destination.read_text())
    payload["weights"][0] = 9.0
    destination.write_text(json.dumps(payload))
    assert load_problem(destination).fingerprint != path_problem.fingerprint


def test_invalid_problem_definitions() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        WeightedGraphProblem("bad", (0.0,), ())
    with pytest.raises(ValueError, match="self-loops"):
        WeightedGraphProblem("bad", (1.0, 2.0), ((0, 0),))
    with pytest.raises(ValueError, match="unique"):
        WeightedGraphProblem("bad", (1.0, 2.0), ((0, 1), (1, 0)))


def test_generator_is_deterministic_and_exposes_shifts() -> None:
    config = GeneratorConfig(vertex_count=10, regime="dense", seed=44)
    first = generate_problem(config)
    second = generate_problem(config)
    assert first == second
    assert len(first.edges) > 15
    lognormal = generate_problem(
        GeneratorConfig(vertex_count=10, regime="weight_lognormal", seed=44)
    )
    assert lognormal.weights != first.weights
    batch = generate_problems(
        count=4,
        min_vertices=5,
        max_vertices=7,
        seed=2,
        regimes=("in_distribution", "clustered"),
    )
    assert len(batch) == 4
    assert {problem.regime for problem in batch} == {"in_distribution", "clustered"}
