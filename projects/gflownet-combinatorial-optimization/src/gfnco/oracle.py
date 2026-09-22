"""Exact independent-set enumeration and reward-proportional target distributions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from gfnco.domain import WeightedGraphProblem


@dataclass(frozen=True, slots=True)
class ExactSolutionSet:
    problem_fingerprint: str
    masks: tuple[int, ...]
    objectives: tuple[float, ...]
    optimum_objective: float
    optimum_masks: tuple[int, ...]
    runtime_seconds: float

    def __post_init__(self) -> None:
        if len(self.masks) != len(self.objectives):
            raise ValueError("masks and objectives must have equal length")
        if not self.masks:
            raise ValueError("the empty independent-set family is impossible")

    @property
    def independent_set_count(self) -> int:
        return len(self.masks)

    def objective_by_mask(self) -> dict[int, float]:
        return dict(zip(self.masks, self.objectives, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_fingerprint": self.problem_fingerprint,
            "independent_set_count": self.independent_set_count,
            "optimum_objective": self.optimum_objective,
            "optimum_masks": list(self.optimum_masks),
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True, slots=True)
class TargetDistribution:
    beta: float
    masks: tuple[int, ...]
    objectives: tuple[float, ...]
    probabilities: tuple[float, ...]
    log_partition: float
    expected_objective: float
    entropy: float

    def __post_init__(self) -> None:
        if not (len(self.masks) == len(self.objectives) == len(self.probabilities)):
            raise ValueError("target distribution arrays must have equal length")
        total = float(sum(self.probabilities))
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("target probabilities must sum to one")

    def probability_by_mask(self) -> dict[int, float]:
        return dict(zip(self.masks, self.probabilities, strict=True))

    def sample(self, rng: np.random.Generator, sample_count: int) -> tuple[int, ...]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        indices = rng.choice(
            len(self.masks),
            size=sample_count,
            replace=True,
            p=np.asarray(self.probabilities, dtype=float),
        )
        return tuple(self.masks[int(index)] for index in indices.tolist())

    def to_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "support_size": len(self.masks),
            "log_partition": self.log_partition,
            "expected_objective": self.expected_objective,
            "entropy": self.entropy,
        }


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def enumerate_independent_sets(
    problem: WeightedGraphProblem,
    *,
    max_vertices: int = 24,
) -> ExactSolutionSet:
    """Enumerate every independent set by deterministic include/exclude recursion."""

    if problem.vertex_count > max_vertices:
        raise ValueError(
            f"exact enumeration is limited to {max_vertices} vertices; "
            f"received {problem.vertex_count}"
        )
    started = time.perf_counter()
    adjacency = problem.adjacency_masks
    masks: list[int] = []

    def visit(vertex: int, selected_mask: int) -> None:
        if vertex == problem.vertex_count:
            masks.append(selected_mask)
            return
        visit(vertex + 1, selected_mask)
        if adjacency[vertex] & selected_mask == 0:
            visit(vertex + 1, selected_mask | (1 << vertex))

    visit(0, 0)
    masks.sort()
    objectives = tuple(problem.objective(mask) for mask in masks)
    optimum = max(objectives)
    tolerance = 1e-10 * max(1.0, abs(optimum))
    optimum_masks = tuple(
        mask
        for mask, objective in zip(masks, objectives, strict=True)
        if abs(objective - optimum) <= tolerance
    )
    if any(not problem.is_independent(mask) for mask in masks):
        raise RuntimeError("exact enumerator produced an infeasible state")
    return ExactSolutionSet(
        problem_fingerprint=problem.fingerprint,
        masks=tuple(masks),
        objectives=objectives,
        optimum_objective=optimum,
        optimum_masks=optimum_masks,
        runtime_seconds=time.perf_counter() - started,
    )


def build_target_distribution(
    problem: WeightedGraphProblem,
    exact: ExactSolutionSet,
    beta: float,
) -> TargetDistribution:
    if exact.problem_fingerprint != problem.fingerprint:
        raise ValueError("exact solution set belongs to a different problem")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    log_rewards = np.asarray(
        [problem.log_reward(mask, beta) for mask in exact.masks],
        dtype=float,
    )
    log_partition = _logsumexp(log_rewards)
    probabilities_array = np.exp(log_rewards - log_partition)
    probabilities_array /= float(np.sum(probabilities_array))
    objectives = np.asarray(exact.objectives, dtype=float)
    expected_objective = float(np.dot(probabilities_array, objectives))
    positive = probabilities_array > 0.0
    entropy = float(-np.sum(probabilities_array[positive] * np.log(probabilities_array[positive])))
    return TargetDistribution(
        beta=beta,
        masks=exact.masks,
        objectives=exact.objectives,
        probabilities=tuple(float(value) for value in probabilities_array.tolist()),
        log_partition=log_partition,
        expected_objective=expected_objective,
        entropy=entropy,
    )


class ExactFlowPolicy:
    """Exact state-flow recursion for the declared uniform backward policy.

    For a nonterminal independent-set state ``s`` with ``k`` selected vertices,

    ``F(s) = R(s) + sum_v F(s U {v}) / (k + 1)``.

    The first term is the flow of the stop edge. The second term follows from the
    uniform backward probability ``1 / (k + 1)`` at every child state. At the root,
    all trajectory multiplicities cancel and ``F(empty)`` equals the terminal reward
    partition function.
    """

    def __init__(
        self,
        problem: WeightedGraphProblem,
        beta: float,
        *,
        max_vertices: int = 24,
    ) -> None:
        if problem.vertex_count > max_vertices:
            raise ValueError(
                f"exact flow recursion is limited to {max_vertices} vertices; "
                f"received {problem.vertex_count}"
            )
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        self.problem = problem
        self.beta = beta
        self._log_flows: dict[int, float] = {}
        self._compute_log_flow(0)

    @property
    def log_partition(self) -> float:
        return self._log_flows[0]

    @property
    def state_count(self) -> int:
        return len(self._log_flows)

    def _compute_log_flow(self, mask: int) -> float:
        cached = self._log_flows.get(mask)
        if cached is not None:
            return cached
        selected_count = mask.bit_count()
        log_terms = [self.problem.log_reward(mask, self.beta)]
        available = self.problem.available_mask(mask)
        for vertex in range(self.problem.vertex_count):
            if available & (1 << vertex):
                child = mask | (1 << vertex)
                log_terms.append(self._compute_log_flow(child) - math.log(selected_count + 1.0))
        maximum = max(log_terms)
        value = maximum + math.log(sum(math.exp(term - maximum) for term in log_terms))
        self._log_flows[mask] = value
        return value

    def forward_probabilities(self, mask: int) -> dict[int, float]:
        if not self.problem.is_independent(mask):
            raise ValueError("flow policy is defined only on independent-set states")
        if mask not in self._log_flows:
            self._compute_log_flow(mask)
        selected_count = mask.bit_count()
        log_flow = self._log_flows[mask]
        numerators: dict[int, float] = {
            self.problem.vertex_count: self.problem.log_reward(mask, self.beta)
        }
        available = self.problem.available_mask(mask)
        for vertex in range(self.problem.vertex_count):
            if available & (1 << vertex):
                child = mask | (1 << vertex)
                numerators[vertex] = self._compute_log_flow(child) - math.log(selected_count + 1.0)
        probabilities = {
            action: math.exp(log_numerator - log_flow)
            for action, log_numerator in numerators.items()
        }
        total = sum(probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-10):
            raise RuntimeError("exact flow-policy probabilities do not sum to one")
        return probabilities

    def sample(self, rng: np.random.Generator, sample_count: int) -> tuple[int, ...]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        samples: list[int] = []
        stop_action = self.problem.vertex_count
        for _ in range(sample_count):
            mask = 0
            for _ in range(self.problem.vertex_count + 1):
                distribution = self.forward_probabilities(mask)
                actions = tuple(distribution)
                probabilities = np.asarray(
                    [distribution[action] for action in actions],
                    dtype=float,
                )
                action = actions[int(rng.choice(len(actions), p=probabilities))]
                if action == stop_action:
                    samples.append(mask)
                    break
                mask = self.problem.add_vertex(mask, action)
            else:  # pragma: no cover - protected by the acyclic state graph
                raise RuntimeError("exact flow-policy sampler failed to terminate")
        return tuple(samples)


def build_exact_flow_policy(
    problem: WeightedGraphProblem,
    exact: ExactSolutionSet,
    beta: float,
) -> ExactFlowPolicy:
    policy = ExactFlowPolicy(problem, beta)
    target = build_target_distribution(problem, exact, beta)
    if not math.isclose(
        policy.log_partition,
        target.log_partition,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("state-flow partition disagrees with terminal enumeration")
    return policy
