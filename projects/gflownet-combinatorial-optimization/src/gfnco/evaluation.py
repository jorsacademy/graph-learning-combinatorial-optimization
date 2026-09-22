"""Exact-distribution, solution-quality, and diversity evaluation."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from gfnco.baselines import (
    exact_flow_policy_samples,
    random_sequential_samples,
    repeated_greedy_samples,
    reward_biased_sequential_samples,
    target_oracle_samples,
    uniform_exact_samples,
)
from gfnco.domain import WeightedGraphProblem
from gfnco.model import GFlowNetPolicy
from gfnco.oracle import (
    ExactSolutionSet,
    TargetDistribution,
    build_exact_flow_policy,
    build_target_distribution,
    enumerate_independent_sets,
)
from gfnco.trajectory import TrajectoryRecord, rollout_model


@dataclass(frozen=True, slots=True)
class SamplingMetrics:
    method: str
    problem_name: str
    problem_fingerprint: str
    scenario: str
    regime: str
    vertex_count: int
    edge_count: int
    beta: float
    sample_count: int
    support_size: int
    feasible_rate: float
    unique_count: int
    unique_rate: float
    support_coverage: float
    target_mass_covered: float
    empirical_entropy: float
    effective_sample_size: float
    mean_objective: float
    target_expected_objective: float
    best_objective: float
    optimum_objective: float
    mean_gap_percent: float
    best_gap_percent: float
    optimum_hit_rate: float
    near_optimal_mode_coverage: float
    mean_pairwise_hamming: float
    total_variation: float
    jensen_shannon: float
    probability_log_correlation: float | None
    probability_log_slope: float | None
    mean_absolute_tb_residual: float | None
    root_mean_square_tb_residual: float | None
    predicted_log_partition: float | None
    exact_log_partition: float
    log_partition_absolute_error: float | None
    runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[SamplingMetrics, ...]
    aggregate: dict[str, dict[str, float | None]]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "aggregate": self.aggregate,
            "metadata": self.metadata,
        }


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    positive = p > 0.0
    return float(np.sum(p[positive] * (np.log(p[positive]) - np.log(q[positive]))))


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def _log_probability_calibration(
    target: np.ndarray,
    empirical: np.ndarray,
    sample_count: int,
) -> tuple[float | None, float | None]:
    smoothing = 0.5 / float(sample_count + 0.5 * len(empirical))
    x = np.log(target)
    y = np.log(empirical + smoothing)
    x_variance = float(np.var(x))
    y_variance = float(np.var(y))
    if x_variance <= 1e-14 or y_variance <= 1e-14:
        return None, None
    correlation = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.cov(x, y, ddof=0)[0, 1] / x_variance)
    return correlation, slope


def _mean_pairwise_hamming(
    masks: tuple[int, ...],
    vertex_count: int,
    *,
    seed: int,
    maximum_pairs: int = 10_000,
) -> float:
    if len(masks) < 2:
        return 0.0
    total_pairs = len(masks) * (len(masks) - 1) // 2
    if total_pairs <= maximum_pairs:
        distances = [
            (masks[left] ^ masks[right]).bit_count() / float(vertex_count)
            for left in range(len(masks))
            for right in range(left + 1, len(masks))
        ]
        return float(np.mean(distances))
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(maximum_pairs):
        left = int(rng.integers(0, len(masks)))
        right = int(rng.integers(0, len(masks) - 1))
        if right >= left:
            right += 1
        distances.append((masks[left] ^ masks[right]).bit_count() / float(vertex_count))
    return float(np.mean(distances))


def evaluate_samples(
    *,
    method: str,
    problem: WeightedGraphProblem,
    exact: ExactSolutionSet,
    target: TargetDistribution,
    samples: tuple[int, ...],
    runtime_seconds: float,
    seed: int,
    trajectories: tuple[TrajectoryRecord, ...] = (),
    scenario: str = "unspecified",
    predicted_log_partition: float | None = None,
) -> SamplingMetrics:
    if not samples:
        raise ValueError("samples must be nonempty")
    if exact.problem_fingerprint != problem.fingerprint:
        raise ValueError("exact solution set belongs to a different problem")
    support_index = {mask: index for index, mask in enumerate(exact.masks)}
    audits = [problem.is_independent(mask) for mask in samples]
    feasible_rate = float(np.mean(audits))
    if not all(audits):
        raise RuntimeError(f"method {method} generated an infeasible independent set")
    if any(mask not in support_index for mask in samples):
        raise RuntimeError("sample is absent from the exact feasible support")
    counts = Counter(samples)
    empirical = np.zeros(len(exact.masks), dtype=float)
    for mask, count in counts.items():
        empirical[support_index[mask]] = count / float(len(samples))
    target_probabilities = np.asarray(target.probabilities, dtype=float)
    total_variation = 0.5 * float(np.sum(np.abs(empirical - target_probabilities)))
    jensen_shannon = _jensen_shannon(empirical, target_probabilities)
    target_mass_covered = float(np.sum(target_probabilities[empirical > 0.0]))
    positive = empirical > 0.0
    empirical_entropy = float(-np.sum(empirical[positive] * np.log(empirical[positive])))
    effective_sample_size = 1.0 / float(np.sum(empirical * empirical))
    objectives = np.asarray([problem.objective(mask) for mask in samples], dtype=float)
    gaps = np.asarray(
        [problem.relative_optimality_gap(value, exact.optimum_objective) for value in objectives],
        dtype=float,
    )
    optimum_mask_set = set(exact.optimum_masks)
    optimum_hit_rate = float(np.mean([mask in optimum_mask_set for mask in samples]))
    near_optimal_threshold = 0.95 * exact.optimum_objective
    exact_near_optimal = {
        mask
        for mask, objective in zip(exact.masks, exact.objectives, strict=True)
        if objective >= near_optimal_threshold
    }
    sampled_near_optimal = {
        mask for mask in counts if problem.objective(mask) >= near_optimal_threshold
    }
    near_optimal_mode_coverage = len(sampled_near_optimal) / float(max(1, len(exact_near_optimal)))
    correlation, slope = _log_probability_calibration(
        target_probabilities,
        empirical,
        len(samples),
    )
    residuals = np.asarray(
        [trajectory.trajectory_balance_residual for trajectory in trajectories],
        dtype=float,
    )
    mean_absolute_residual = float(np.mean(np.abs(residuals))) if residuals.size else None
    root_mean_square_residual = (
        float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else None
    )
    return SamplingMetrics(
        method=method,
        problem_name=problem.name,
        problem_fingerprint=problem.fingerprint,
        scenario=scenario,
        regime=problem.regime,
        vertex_count=problem.vertex_count,
        edge_count=len(problem.edges),
        beta=target.beta,
        sample_count=len(samples),
        support_size=len(exact.masks),
        feasible_rate=feasible_rate,
        unique_count=len(counts),
        unique_rate=len(counts) / float(len(samples)),
        support_coverage=len(counts) / float(len(exact.masks)),
        target_mass_covered=target_mass_covered,
        empirical_entropy=empirical_entropy,
        effective_sample_size=effective_sample_size,
        mean_objective=float(np.mean(objectives)),
        target_expected_objective=target.expected_objective,
        best_objective=float(np.max(objectives)),
        optimum_objective=exact.optimum_objective,
        mean_gap_percent=float(np.mean(gaps)),
        best_gap_percent=float(np.min(gaps)),
        optimum_hit_rate=optimum_hit_rate,
        near_optimal_mode_coverage=near_optimal_mode_coverage,
        mean_pairwise_hamming=_mean_pairwise_hamming(
            samples,
            problem.vertex_count,
            seed=seed,
        ),
        total_variation=total_variation,
        jensen_shannon=jensen_shannon,
        probability_log_correlation=correlation,
        probability_log_slope=slope,
        mean_absolute_tb_residual=mean_absolute_residual,
        root_mean_square_tb_residual=root_mean_square_residual,
        predicted_log_partition=predicted_log_partition,
        exact_log_partition=target.log_partition,
        log_partition_absolute_error=(
            abs(predicted_log_partition - target.log_partition)
            if predicted_log_partition is not None
            else None
        ),
        runtime_seconds=runtime_seconds,
    )


def _sample_model_with_trajectories(
    model: GFlowNetPolicy,
    problem: WeightedGraphProblem,
    beta: float,
    *,
    sample_count: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[TrajectoryRecord, ...], float]:
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(seed)
    samples: list[int] = []
    records: list[TrajectoryRecord] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for _ in range(sample_count):
            trajectory = rollout_model(model, problem, beta, generator=generator)
            samples.append(trajectory.terminal_mask)
            records.append(trajectory.detached())
    return tuple(samples), tuple(records), time.perf_counter() - started


def evaluate_problem(
    problem: WeightedGraphProblem,
    *,
    beta: float,
    sample_count: int,
    seed: int,
    gflownet_model: GFlowNetPolicy | None = None,
    reinforce_model: GFlowNetPolicy | None = None,
    max_exact_vertices: int = 24,
    include_oracle_controls: bool = True,
    scenario: str = "unspecified",
) -> tuple[SamplingMetrics, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    exact = enumerate_independent_sets(problem, max_vertices=max_exact_vertices)
    target = build_target_distribution(problem, exact, beta)
    exact_flow_policy = build_exact_flow_policy(problem, exact, beta)
    rows: list[SamplingMetrics] = []

    if gflownet_model is not None:
        samples, trajectories, runtime = _sample_model_with_trajectories(
            gflownet_model,
            problem,
            beta,
            sample_count=sample_count,
            seed=seed + 11,
        )
        rows.append(
            evaluate_samples(
                method="gflownet_tb",
                problem=problem,
                exact=exact,
                target=target,
                samples=samples,
                runtime_seconds=runtime,
                seed=seed + 11,
                trajectories=trajectories,
                scenario=scenario,
                predicted_log_partition=float(
                    gflownet_model.log_partition(problem, beta).detach().cpu()
                ),
            )
        )
    if reinforce_model is not None:
        samples, _trajectories, runtime = _sample_model_with_trajectories(
            reinforce_model,
            problem,
            beta,
            sample_count=sample_count,
            seed=seed + 23,
        )
        rows.append(
            evaluate_samples(
                method="reinforce",
                problem=problem,
                exact=exact,
                target=target,
                samples=samples,
                runtime_seconds=runtime,
                seed=seed + 23,
                scenario=scenario,
            )
        )

    baseline_specs: tuple[tuple[str, Callable[[], tuple[int, ...]]], ...] = (
        (
            "reward_biased_sequential",
            lambda: reward_biased_sequential_samples(
                problem,
                beta,
                sample_count=sample_count,
                seed=seed + 31,
            ),
        ),
        (
            "random_sequential",
            lambda: random_sequential_samples(
                problem,
                sample_count=sample_count,
                seed=seed + 41,
            ),
        ),
        (
            "greedy_weight_degree",
            lambda: repeated_greedy_samples(problem, sample_count=sample_count),
        ),
    )
    for label, sampler in baseline_specs:
        started = time.perf_counter()
        samples = sampler()
        runtime = time.perf_counter() - started
        rows.append(
            evaluate_samples(
                method=label,
                problem=problem,
                exact=exact,
                target=target,
                samples=samples,
                runtime_seconds=runtime,
                seed=seed,
                scenario=scenario,
            )
        )

    if include_oracle_controls:
        oracle_specs: tuple[tuple[str, Callable[[], tuple[int, ...]]], ...] = (
            (
                "exact_flow_policy_oracle",
                lambda: exact_flow_policy_samples(
                    exact_flow_policy,
                    sample_count=sample_count,
                    seed=seed + 47,
                ),
            ),
            (
                "uniform_exact_oracle",
                lambda: uniform_exact_samples(
                    exact,
                    sample_count=sample_count,
                    seed=seed + 53,
                ),
            ),
            (
                "target_distribution_oracle",
                lambda: target_oracle_samples(
                    target,
                    sample_count=sample_count,
                    seed=seed + 67,
                ),
            ),
        )
        for label, sampler in oracle_specs:
            started = time.perf_counter()
            samples = sampler()
            runtime = time.perf_counter() - started
            rows.append(
                evaluate_samples(
                    method=label,
                    problem=problem,
                    exact=exact,
                    target=target,
                    samples=samples,
                    runtime_seconds=runtime,
                    seed=seed,
                    scenario=scenario,
                )
            )
    return tuple(rows)


def _optional_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def aggregate_rows(rows: tuple[SamplingMetrics, ...]) -> dict[str, dict[str, float | None]]:
    groups = sorted({(row.scenario, row.method) for row in rows})
    aggregate: dict[str, dict[str, float | None]] = {}
    for scenario, method in groups:
        selected = [row for row in rows if row.scenario == scenario and row.method == method]
        aggregate[f"{scenario}|{method}"] = {
            "instances": float(len(selected)),
            "feasible_rate": statistics.fmean(row.feasible_rate for row in selected),
            "mean_total_variation": statistics.fmean(row.total_variation for row in selected),
            "mean_jensen_shannon": statistics.fmean(row.jensen_shannon for row in selected),
            "mean_target_mass_covered": statistics.fmean(
                row.target_mass_covered for row in selected
            ),
            "mean_objective": statistics.fmean(row.mean_objective for row in selected),
            "mean_gap_percent": statistics.fmean(row.mean_gap_percent for row in selected),
            "mean_optimum_hit_rate": statistics.fmean(row.optimum_hit_rate for row in selected),
            "mean_near_optimal_mode_coverage": statistics.fmean(
                row.near_optimal_mode_coverage for row in selected
            ),
            "mean_pairwise_hamming": statistics.fmean(
                row.mean_pairwise_hamming for row in selected
            ),
            "mean_unique_rate": statistics.fmean(row.unique_rate for row in selected),
            "mean_probability_log_correlation": _optional_mean(
                [row.probability_log_correlation for row in selected]
            ),
            "mean_absolute_tb_residual": _optional_mean(
                [row.mean_absolute_tb_residual for row in selected]
            ),
            "mean_log_partition_absolute_error": _optional_mean(
                [row.log_partition_absolute_error for row in selected]
            ),
            "mean_runtime_seconds": statistics.fmean(row.runtime_seconds for row in selected),
        }
    return aggregate


def build_benchmark_report(
    rows: tuple[SamplingMetrics, ...],
    *,
    metadata: dict[str, object] | None = None,
) -> BenchmarkReport:
    if not rows:
        raise ValueError("benchmark rows must be nonempty")
    return BenchmarkReport(
        rows=rows,
        aggregate=aggregate_rows(rows),
        metadata=dict(metadata or {}),
    )


def save_report_json(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in report.rows]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
