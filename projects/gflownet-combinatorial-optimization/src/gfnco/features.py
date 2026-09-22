"""Permutation-equivariant graph/state features for the constructive policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import ConstructionState

FEATURE_SCHEMA_VERSION = "1.0"
NODE_FEATURE_NAMES = (
    "weight_fraction",
    "weight_to_max",
    "degree_fraction",
    "inverse_degree",
    "selected",
    "available",
    "blocked",
    "bias",
)
GLOBAL_FEATURE_NAMES = (
    "log_vertex_count",
    "edge_density",
    "weight_coefficient_of_variation",
    "selected_fraction",
    "available_fraction",
    "beta_scaled",
)


@dataclass(frozen=True, slots=True)
class GraphStateFeatures:
    node_features: Tensor
    adjacency: Tensor
    normalized_adjacency: Tensor
    global_features: Tensor
    valid_action_mask: Tensor

    @property
    def vertex_count(self) -> int:
        return int(self.node_features.shape[0])


def feature_schema() -> dict[str, object]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "node_features": list(NODE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "action_space": "one action per vertex plus a dedicated stop action",
    }


def featurize_state(
    problem: WeightedGraphProblem,
    state: ConstructionState,
    beta: float,
    *,
    beta_scale: float = 10.0,
    device: torch.device | str = "cpu",
) -> GraphStateFeatures:
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    if beta_scale <= 0.0 or not math.isfinite(beta_scale):
        raise ValueError("beta_scale must be finite and positive")
    n = problem.vertex_count
    weights = np.asarray(problem.weights, dtype=np.float32)
    degrees = np.asarray(problem.degrees, dtype=np.float32)
    weight_fraction = weights / float(problem.total_weight)
    weight_to_max = weights / float(np.max(weights))
    degree_denominator = float(max(1, n - 1))
    degree_fraction = degrees / degree_denominator
    inverse_degree = 1.0 / (1.0 + degrees)

    selected = np.asarray(
        [float(bool(state.selected_mask & (1 << vertex))) for vertex in range(n)],
        dtype=np.float32,
    )
    available_mask = 0 if state.stopped else problem.available_mask(state.selected_mask)
    available = np.asarray(
        [float(bool(available_mask & (1 << vertex))) for vertex in range(n)],
        dtype=np.float32,
    )
    blocked = 1.0 - selected - available
    blocked = np.maximum(blocked, 0.0)
    node_features = np.column_stack(
        [
            weight_fraction,
            weight_to_max,
            degree_fraction,
            inverse_degree,
            selected,
            available,
            blocked,
            np.ones(n, dtype=np.float32),
        ]
    )

    adjacency = np.zeros((n, n), dtype=np.float32)
    for u, v in problem.edges:
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    row_sums = np.sum(adjacency, axis=1, keepdims=True)
    normalized = adjacency / np.maximum(row_sums, 1.0)
    edge_density = 2.0 * len(problem.edges) / float(max(1, n * (n - 1)))
    mean_weight = float(np.mean(weights))
    coefficient_of_variation = float(np.std(weights) / max(mean_weight, 1e-8))
    global_features = np.asarray(
        [
            math.log1p(n) / math.log(129.0),
            edge_density,
            coefficient_of_variation,
            state.selected_mask.bit_count() / float(n),
            available_mask.bit_count() / float(n),
            beta / beta_scale,
        ],
        dtype=np.float32,
    )
    valid_actions = np.zeros(n + 1, dtype=bool)
    if not state.stopped:
        for vertex in range(n):
            valid_actions[vertex] = bool(available_mask & (1 << vertex))
        valid_actions[n] = True
    target_device = torch.device(device)
    return GraphStateFeatures(
        node_features=torch.as_tensor(node_features, device=target_device),
        adjacency=torch.as_tensor(adjacency, device=target_device),
        normalized_adjacency=torch.as_tensor(normalized, device=target_device),
        global_features=torch.as_tensor(global_features, device=target_device),
        valid_action_mask=torch.as_tensor(valid_actions, device=target_device),
    )
