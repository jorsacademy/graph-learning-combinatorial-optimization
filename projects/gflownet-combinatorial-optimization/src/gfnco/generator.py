"""Deterministic synthetic weighted-graph instance generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gfnco.domain import WeightedGraphProblem

GraphRegime = Literal[
    "in_distribution",
    "sparse",
    "dense",
    "clustered",
    "weight_lognormal",
    "weight_bimodal",
    "degree_correlated",
    "combined_shift",
]

SUPPORTED_REGIMES: tuple[GraphRegime, ...] = (
    "in_distribution",
    "sparse",
    "dense",
    "clustered",
    "weight_lognormal",
    "weight_bimodal",
    "degree_correlated",
    "combined_shift",
)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    vertex_count: int = 12
    edge_probability: float = 0.30
    regime: GraphRegime = "in_distribution"
    seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.vertex_count <= 128:
            raise ValueError("vertex_count must lie between 2 and 128")
        if not 0.0 <= self.edge_probability <= 1.0:
            raise ValueError("edge_probability must lie in [0, 1]")
        if self.regime not in SUPPORTED_REGIMES:
            raise ValueError(f"unsupported graph regime: {self.regime}")


def _edge_probability(config: GeneratorConfig) -> float:
    if config.regime == "sparse":
        return min(config.edge_probability, 0.12)
    if config.regime == "dense":
        return max(config.edge_probability, 0.62)
    if config.regime == "combined_shift":
        return max(config.edge_probability, 0.56)
    return config.edge_probability


def _generate_edges(
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    n = config.vertex_count
    edges: list[tuple[int, int]] = []
    if config.regime == "clustered":
        split = max(1, n // 2)
        for u in range(n):
            for v in range(u + 1, n):
                same_cluster = (u < split) == (v < split)
                probability = 0.58 if same_cluster else 0.08
                if rng.random() < probability:
                    edges.append((u, v))
    else:
        probability = _edge_probability(config)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < probability:
                    edges.append((u, v))
    return tuple(edges)


def _generate_weights(
    config: GeneratorConfig,
    edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[float, ...]:
    n = config.vertex_count
    if config.regime in {"weight_lognormal", "combined_shift"}:
        values = rng.lognormal(mean=1.4, sigma=0.65, size=n)
    elif config.regime == "weight_bimodal":
        high = rng.random(n) < 0.30
        values = np.where(high, rng.uniform(8.0, 14.0, n), rng.uniform(1.0, 5.0, n))
    elif config.regime == "degree_correlated":
        degrees = np.zeros(n, dtype=float)
        for u, v in edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        noise = rng.uniform(0.5, 2.5, n)
        values = 1.5 + 1.8 * degrees + noise
    else:
        values = rng.uniform(1.0, 10.0, n)
    return tuple(float(round(value, 6)) for value in values)


def generate_problem(config: GeneratorConfig | None = None) -> WeightedGraphProblem:
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)
    edges = _generate_edges(config, rng)
    weights = _generate_weights(config, edges, rng)
    return WeightedGraphProblem(
        name=(
            f"mwis-n{config.vertex_count}-{config.regime}-"
            f"p{_edge_probability(config):.2f}-seed{config.seed}"
        ),
        weights=weights,
        edges=edges,
        regime=config.regime,
        seed=config.seed,
        metadata={
            "generator": "gfnco-erdos-renyi-v1",
            "edge_probability": _edge_probability(config),
        },
    )


def generate_problems(
    *,
    count: int,
    min_vertices: int,
    max_vertices: int,
    seed: int,
    regimes: tuple[GraphRegime, ...] = ("in_distribution",),
    edge_probability: float = 0.30,
) -> tuple[WeightedGraphProblem, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if min_vertices < 2 or max_vertices < min_vertices:
        raise ValueError("invalid vertex-count range")
    if not regimes:
        raise ValueError("at least one regime is required")
    if any(regime not in SUPPORTED_REGIMES for regime in regimes):
        raise ValueError("unsupported regime in regimes")
    rng = np.random.default_rng(seed)
    problems: list[WeightedGraphProblem] = []
    for index in range(count):
        vertex_count = int(rng.integers(min_vertices, max_vertices + 1))
        regime = regimes[index % len(regimes)]
        instance_seed = seed + 104_729 * (index + 1)
        problems.append(
            generate_problem(
                GeneratorConfig(
                    vertex_count=vertex_count,
                    edge_probability=edge_probability,
                    regime=regime,
                    seed=instance_seed,
                )
            )
        )
    return tuple(problems)
