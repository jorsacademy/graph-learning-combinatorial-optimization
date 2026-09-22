from __future__ import annotations

import torch

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import ConstructionState
from gfnco.features import (
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    feature_schema,
    featurize_state,
)
from gfnco.model import GFlowNetPolicy, ModelConfig, clone_model, load_checkpoint, save_checkpoint


def _permute_problem(
    problem: WeightedGraphProblem,
    new_to_old: tuple[int, ...],
) -> tuple[WeightedGraphProblem, tuple[int, ...]]:
    old_to_new = [0] * problem.vertex_count
    for new_index, old_index in enumerate(new_to_old):
        old_to_new[old_index] = new_index
    edges = tuple((old_to_new[u], old_to_new[v]) for u, v in problem.edges)
    permuted = WeightedGraphProblem(
        name="permuted",
        weights=tuple(problem.weights[index] for index in new_to_old),
        edges=edges,
    )
    return permuted, tuple(old_to_new)


def test_feature_shapes_and_action_mask(path_problem: WeightedGraphProblem) -> None:
    state = ConstructionState(selected_mask=1)
    features = featurize_state(path_problem, state, beta=4.0)
    assert features.node_features.shape == (3, len(NODE_FEATURE_NAMES))
    assert features.global_features.shape == (len(GLOBAL_FEATURE_NAMES),)
    assert features.adjacency.shape == (3, 3)
    assert features.valid_action_mask.tolist() == [False, False, True, True]
    assert feature_schema()["node_features"] == list(NODE_FEATURE_NAMES)


def test_model_masks_invalid_actions(path_problem: WeightedGraphProblem) -> None:
    model = GFlowNetPolicy(ModelConfig(hidden_dim=16, message_passing_rounds=2))
    logits = model.action_logits(path_problem, ConstructionState(selected_mask=1), beta=4.0)
    assert logits.shape == (4,)
    assert logits[0] < -1e20
    assert logits[1] < -1e20
    assert torch.isfinite(logits[2:]).all()
    assert torch.isfinite(model.log_partition(path_problem, 4.0))
    assert model.parameter_count > 0


def test_policy_is_permutation_equivariant() -> None:
    problem = WeightedGraphProblem(
        "graph",
        (1.0, 4.0, 2.0, 3.0),
        ((0, 1), (1, 2), (2, 3)),
    )
    new_to_old = (2, 0, 3, 1)
    permuted, old_to_new = _permute_problem(problem, new_to_old)
    original_mask = (1 << 0) | (1 << 3)
    permuted_mask = sum(1 << old_to_new[index] for index in (0, 3))
    torch.manual_seed(5)
    model = GFlowNetPolicy(ModelConfig(hidden_dim=16, message_passing_rounds=2))
    model.eval()
    with torch.no_grad():
        original_logits = model.action_logits(
            problem,
            ConstructionState(original_mask),
            beta=5.0,
        )
        permuted_logits = model.action_logits(
            permuted,
            ConstructionState(permuted_mask),
            beta=5.0,
        )
    for new_index, old_index in enumerate(new_to_old):
        assert torch.allclose(permuted_logits[new_index], original_logits[old_index], atol=1e-6)
    assert torch.allclose(permuted_logits[-1], original_logits[-1], atol=1e-6)


def test_safe_checkpoint_round_trip(tmp_path, path_problem: WeightedGraphProblem) -> None:
    torch.manual_seed(6)
    model = GFlowNetPolicy(ModelConfig(hidden_dim=10, message_passing_rounds=1))
    destination = tmp_path / "model.safetensors"
    save_checkpoint(model, destination, metadata={"algorithm": "trajectory_balance"})
    loaded, metadata = load_checkpoint(destination)
    assert metadata == {"algorithm": "trajectory_balance"}
    state = ConstructionState(selected_mask=0)
    with torch.no_grad():
        assert torch.allclose(
            model.action_logits(path_problem, state, 3.0),
            loaded.action_logits(path_problem, state, 3.0),
        )
    clone = clone_model(model)
    with torch.no_grad():
        assert torch.allclose(
            model.log_partition(path_problem, 3.0),
            clone.log_partition(path_problem, 3.0),
        )
