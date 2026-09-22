from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest

from gfnco.baselines import repeated_greedy_samples
from gfnco.domain import WeightedGraphProblem
from gfnco.evaluation import (
    build_benchmark_report,
    evaluate_problem,
    evaluate_samples,
    save_report_csv,
    save_report_json,
)
from gfnco.model import GFlowNetPolicy, ModelConfig
from gfnco.oracle import build_target_distribution, enumerate_independent_sets


def test_exact_target_sampler_recovers_target_distribution(
    path_problem: WeightedGraphProblem,
) -> None:
    exact = enumerate_independent_sets(path_problem)
    target = build_target_distribution(path_problem, exact, beta=5.0)
    samples = target.sample(np.random.default_rng(7), 8_000)
    metrics = evaluate_samples(
        method="target",
        problem=path_problem,
        exact=exact,
        target=target,
        samples=samples,
        runtime_seconds=0.0,
        seed=7,
        scenario="unit",
    )
    assert metrics.feasible_rate == 1.0
    assert metrics.total_variation < 0.03
    assert metrics.jensen_shannon < 0.01
    assert metrics.target_mass_covered == 1.0
    assert metrics.probability_log_correlation is not None
    assert metrics.probability_log_correlation > 0.95


def test_greedy_control_is_feasible_but_not_diverse(path_problem: WeightedGraphProblem) -> None:
    exact = enumerate_independent_sets(path_problem)
    target = build_target_distribution(path_problem, exact, beta=4.0)
    samples = repeated_greedy_samples(path_problem, sample_count=20)
    metrics = evaluate_samples(
        method="greedy",
        problem=path_problem,
        exact=exact,
        target=target,
        samples=samples,
        runtime_seconds=0.0,
        seed=1,
    )
    assert metrics.feasible_rate == 1.0
    assert metrics.unique_count == 1
    assert metrics.unique_rate == 0.05
    assert metrics.mean_pairwise_hamming == 0.0


def test_evaluate_problem_and_report_outputs(tmp_path, path_problem: WeightedGraphProblem) -> None:
    gflownet = GFlowNetPolicy(ModelConfig(hidden_dim=10, message_passing_rounds=1))
    reinforce = GFlowNetPolicy(ModelConfig(hidden_dim=10, message_passing_rounds=1))
    rows = evaluate_problem(
        path_problem,
        beta=4.0,
        sample_count=30,
        seed=10,
        gflownet_model=gflownet,
        reinforce_model=reinforce,
        scenario="tiny",
    )
    methods = {row.method for row in rows}
    assert {
        "gflownet_tb",
        "reinforce",
        "reward_biased_sequential",
        "random_sequential",
        "greedy_weight_degree",
        "exact_flow_policy_oracle",
        "uniform_exact_oracle",
        "target_distribution_oracle",
    } == methods
    assert all(row.scenario == "tiny" for row in rows)
    assert all(row.feasible_rate == 1.0 for row in rows)
    report = build_benchmark_report(rows, metadata={"test": True})
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json.loads(json_path.read_text())["metadata"] == {"test": True}
    with csv_path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    assert "tiny|gflownet_tb" in report.aggregate


def test_evaluation_fails_closed_on_infeasible_sample(path_problem: WeightedGraphProblem) -> None:
    exact = enumerate_independent_sets(path_problem)
    target = build_target_distribution(path_problem, exact, beta=1.0)
    with pytest.raises(RuntimeError, match="infeasible"):
        evaluate_samples(
            method="bad",
            problem=path_problem,
            exact=exact,
            target=target,
            samples=(0b011,),
            runtime_seconds=time.perf_counter() - time.perf_counter(),
            seed=1,
        )
