"""Typed maximum-weight independent-set problem definitions and audits."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")
    return float(value)


def _require_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a JSON integer")
    return value


@dataclass(frozen=True, slots=True)
class FeasibilityAudit:
    """Independent verification of a candidate binary decision."""

    feasible: bool
    integrality_violation: float
    bound_violation: float
    conflicting_edges: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "integrality_violation": self.integrality_violation,
            "bound_violation": self.bound_violation,
            "conflicting_edges": [list(edge) for edge in self.conflicting_edges],
        }


@dataclass(frozen=True, slots=True)
class WeightedGraphProblem:
    """A finite maximum-weight independent-set instance.

    All vertex weights are strictly positive. This keeps the target reward monotone in
    solution value and makes the empty set a valid, strictly positive-reward terminal.
    """

    name: str
    weights: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]
    regime: str = "in_distribution"
    seed: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("problem name must be nonempty")
        if not self.weights:
            raise ValueError("at least one vertex is required")
        if len(self.weights) > 128:
            raise ValueError("at most 128 vertices are supported by the reference implementation")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in self.weights):
            raise ValueError("all vertex weights must be finite and strictly positive")
        canonical_edges: list[tuple[int, int]] = []
        vertex_count = len(self.weights)
        for raw_u, raw_v in self.edges:
            if raw_u == raw_v:
                raise ValueError("self-loops are not supported")
            if not (0 <= raw_u < vertex_count and 0 <= raw_v < vertex_count):
                raise ValueError("edge endpoint is outside the vertex range")
            canonical_edges.append((min(raw_u, raw_v), max(raw_u, raw_v)))
        canonical = tuple(sorted(set(canonical_edges)))
        if len(canonical) != len(canonical_edges):
            raise ValueError("edges must be unique")
        object.__setattr__(self, "edges", canonical)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def vertex_count(self) -> int:
        return len(self.weights)

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights))

    @property
    def full_mask(self) -> int:
        return (1 << self.vertex_count) - 1

    @property
    def adjacency_masks(self) -> tuple[int, ...]:
        masks = [0] * self.vertex_count
        for u, v in self.edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        return tuple(masks)

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(mask.bit_count() for mask in self.adjacency_masks)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def selected_vertices(self, mask: int) -> tuple[int, ...]:
        self._validate_mask_range(mask)
        return tuple(index for index in range(self.vertex_count) if mask & (1 << index))

    def objective(self, mask: int) -> float:
        return float(sum(self.weights[index] for index in self.selected_vertices(mask)))

    def is_independent(self, mask: int) -> bool:
        self._validate_mask_range(mask)
        adjacency = self.adjacency_masks
        remaining = mask
        while remaining:
            least_bit = remaining & -remaining
            vertex = least_bit.bit_length() - 1
            if adjacency[vertex] & mask:
                return False
            remaining ^= least_bit
        return True

    def available_mask(self, selected_mask: int) -> int:
        """Return vertices that can be added without violating independence."""

        self._validate_mask_range(selected_mask)
        blocked = selected_mask
        adjacency = self.adjacency_masks
        remaining = selected_mask
        while remaining:
            least_bit = remaining & -remaining
            vertex = least_bit.bit_length() - 1
            blocked |= adjacency[vertex]
            remaining ^= least_bit
        return self.full_mask & ~blocked

    def add_vertex(self, selected_mask: int, vertex: int) -> int:
        self._validate_mask_range(selected_mask)
        if not 0 <= vertex < self.vertex_count:
            raise ValueError("vertex is outside the valid range")
        if not (self.available_mask(selected_mask) & (1 << vertex)):
            raise ValueError("vertex is not a valid forward action")
        return selected_mask | (1 << vertex)

    def mask_to_decision(self, mask: int) -> tuple[int, ...]:
        self._validate_mask_range(mask)
        return tuple(int(bool(mask & (1 << index))) for index in range(self.vertex_count))

    def decision_to_mask(
        self,
        decision: tuple[float, ...] | list[float] | np.ndarray,
        *,
        tolerance: float = 1e-8,
    ) -> int:
        values = np.asarray(decision, dtype=float)
        if values.shape != (self.vertex_count,):
            raise ValueError("decision has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("decision values must be finite")
        rounded = np.rint(values)
        if float(np.max(np.abs(values - rounded))) > tolerance:
            raise ValueError("decision is not binary within tolerance")
        if float(np.max(np.maximum(0.0, -values))) > tolerance:
            raise ValueError("decision violates the lower binary bound")
        if float(np.max(np.maximum(0.0, values - 1.0))) > tolerance:
            raise ValueError("decision violates the upper binary bound")
        mask = 0
        for index, value in enumerate(rounded.astype(int).tolist()):
            if value == 1:
                mask |= 1 << index
        return mask

    def audit_decision(
        self,
        decision: tuple[float, ...] | list[float] | np.ndarray,
        *,
        tolerance: float = 1e-8,
    ) -> FeasibilityAudit:
        values = np.asarray(decision, dtype=float)
        if values.shape != (self.vertex_count,):
            raise ValueError("decision has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("decision values must be finite")
        rounded = np.rint(values)
        integrality_violation = float(np.max(np.abs(values - rounded)))
        lower_violation = float(np.max(np.maximum(0.0, -values)))
        upper_violation = float(np.max(np.maximum(0.0, values - 1.0)))
        bound_violation = max(lower_violation, upper_violation)
        selected = {index for index, value in enumerate(rounded.tolist()) if int(value) == 1}
        conflicts = tuple(
            edge for edge in self.edges if edge[0] in selected and edge[1] in selected
        )
        feasible = (
            integrality_violation <= tolerance and bound_violation <= tolerance and not conflicts
        )
        return FeasibilityAudit(
            feasible=feasible,
            integrality_violation=integrality_violation,
            bound_violation=bound_violation,
            conflicting_edges=conflicts,
        )

    def log_reward(self, mask: int, beta: float) -> float:
        """Bounded log reward used by the target distribution.

        ``log R(x) = beta * value(x) / sum_i w_i`` lies in ``[0, beta]``.
        """

        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        if not self.is_independent(mask):
            raise ValueError("reward is defined only for feasible independent sets")
        return beta * self.objective(mask) / self.total_weight

    def relative_optimality_gap(self, candidate: float, optimum: float) -> float:
        if candidate > optimum + 1e-8 * max(1.0, abs(optimum)):
            raise RuntimeError("candidate objective exceeds the exact optimum beyond tolerance")
        return max(0.0, 100.0 * (optimum - candidate) / max(1.0, abs(optimum)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weights": list(self.weights),
            "edges": [list(edge) for edge in self.edges],
            "regime": self.regime,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WeightedGraphProblem:
        raw_weights = cast(list[object], payload["weights"])
        weights = tuple(_require_number(value, "weight") for value in raw_weights)
        raw_edges = cast(list[object], payload["edges"])
        edges: list[tuple[int, int]] = []
        for raw_edge in raw_edges:
            edge = cast(list[object], raw_edge)
            if len(edge) != 2:
                raise ValueError("each edge must contain exactly two endpoints")
            edges.append(
                (
                    _require_integer(edge[0], "edge endpoint"),
                    _require_integer(edge[1], "edge endpoint"),
                )
            )
        raw_seed = payload.get("seed", 0)
        return cls(
            name=str(payload["name"]),
            weights=weights,
            edges=tuple(edges),
            regime=str(payload.get("regime", "in_distribution")),
            seed=_require_integer(raw_seed, "seed"),
            metadata=cast(dict[str, object], payload.get("metadata", {})),
        )

    def _validate_mask_range(self, mask: int) -> None:
        if mask < 0 or mask > self.full_mask:
            raise ValueError("mask is outside the valid state space")


def save_problem(problem: WeightedGraphProblem, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(problem.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_problem(path: str | Path) -> WeightedGraphProblem:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("problem file must contain a JSON object")
    return WeightedGraphProblem.from_dict(cast(dict[str, object], payload))
