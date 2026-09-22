"""A compact instance-conditioned graph policy for GFlowNet and RL controls."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import ConstructionState
from gfnco.features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    featurize_state,
)

CHECKPOINT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int = 64
    message_passing_rounds: int = 3
    beta_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.message_passing_rounds <= 0:
            raise ValueError("message_passing_rounds must be positive")
        if self.beta_scale <= 0.0:
            raise ValueError("beta_scale must be positive")


class GFlowNetPolicy(nn.Module):
    """Shared graph encoder with forward-policy and log-partition heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(len(NODE_FEATURE_NAMES), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURE_NAMES), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_layers = nn.ModuleList(
            [nn.Linear(3 * hidden, hidden) for _ in range(self.config.message_passing_rounds)]
        )
        self.message_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(self.config.message_passing_rounds)]
        )
        self.graph_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.node_action_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.log_partition_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(
        self,
        problem: WeightedGraphProblem,
        state: ConstructionState,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = featurize_state(
            problem,
            state,
            beta,
            beta_scale=self.config.beta_scale,
            device=self.device,
        )
        node_embeddings = self.node_encoder(features.node_features)
        global_embedding = self.global_encoder(features.global_features)
        for layer, norm in zip(self.message_layers, self.message_norms, strict=True):
            neighbor_embeddings = features.normalized_adjacency @ node_embeddings
            repeated_global = global_embedding.unsqueeze(0).expand(problem.vertex_count, -1)
            delta = torch.tanh(
                layer(torch.cat([node_embeddings, neighbor_embeddings, repeated_global], dim=-1))
            )
            node_embeddings = norm(node_embeddings + delta)
        mean_pool = torch.mean(node_embeddings, dim=0)
        max_pool = torch.max(node_embeddings, dim=0).values
        graph_embedding = self.graph_head(
            torch.cat([mean_pool, max_pool, global_embedding], dim=-1)
        )
        return node_embeddings, graph_embedding, features.valid_action_mask

    def action_logits(
        self,
        problem: WeightedGraphProblem,
        state: ConstructionState,
        beta: float,
    ) -> Tensor:
        if state.stopped:
            raise ValueError("terminal states have no forward-policy logits")
        node_embeddings, graph_embedding, valid_mask = self._encode(problem, state, beta)
        repeated_graph = graph_embedding.unsqueeze(0).expand(problem.vertex_count, -1)
        node_logits = self.node_action_head(
            torch.cat([node_embeddings, repeated_graph], dim=-1)
        ).squeeze(-1)
        stop_logit = self.stop_head(graph_embedding).reshape(1)
        logits = torch.cat([node_logits, stop_logit], dim=0)
        if not torch.any(valid_mask):
            raise RuntimeError("a nonterminal state must expose at least the stop action")
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("model produced non-finite action logits")
        minimum = torch.finfo(logits.dtype).min
        return torch.where(valid_mask, logits, torch.full_like(logits, minimum))

    def log_partition(self, problem: WeightedGraphProblem, beta: float) -> Tensor:
        _, graph_embedding, _ = self._encode(problem, ConstructionState(), beta)
        value: Tensor = self.log_partition_head(graph_embedding).squeeze()
        if not torch.isfinite(value):
            raise RuntimeError("model produced a non-finite log-partition estimate")
        return value


def clone_model(model: GFlowNetPolicy) -> GFlowNetPolicy:
    clone = GFlowNetPolicy(model.config)
    clone.load_state_dict(model.state_dict())
    clone.to(model.device)
    return clone


def save_checkpoint(
    model: GFlowNetPolicy,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
        "metadata": json.dumps(metadata or {}, sort_keys=True),
    }
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(tensors, str(output), metadata=header)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[GFlowNetPolicy, dict[str, object]]:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensor_keys = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in tensor_keys}
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    raw_config = json.loads(header["model_config"])
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = ModelConfig(
        hidden_dim=int(raw_config["hidden_dim"]),
        message_passing_rounds=int(raw_config["message_passing_rounds"]),
        beta_scale=float(raw_config["beta_scale"]),
    )
    model = GFlowNetPolicy(config)
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, cast(dict[str, object], raw_metadata)
