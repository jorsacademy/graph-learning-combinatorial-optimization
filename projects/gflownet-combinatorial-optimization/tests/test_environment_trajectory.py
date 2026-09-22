from __future__ import annotations

import math

import torch

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import ConstructionState, IndependentSetEnvironment
from gfnco.model import GFlowNetPolicy, ModelConfig
from gfnco.trajectory import greedy_model_mask, rollout_model, sample_model_masks


def test_environment_forward_and_backward_accounting(
    path_problem: WeightedGraphProblem,
) -> None:
    environment = IndependentSetEnvironment(path_problem)
    state = environment.initial_state
    assert environment.valid_actions(state) == (0, 1, 2, 3)
    first = environment.transition(state, 0)
    assert first.current.selected_mask == 1
    assert first.backward_log_probability == 0.0
    second = environment.transition(first.current, 2)
    assert second.current.selected_mask == 5
    assert math.isclose(second.backward_log_probability, -math.log(2.0))
    terminal = environment.transition(second.current, environment.stop_action)
    assert terminal.current.stopped
    assert environment.terminal_mask(terminal.current) == 5


def test_rollout_is_feasible_and_tb_residual_matches_formula(
    path_problem: WeightedGraphProblem,
) -> None:
    model = GFlowNetPolicy(ModelConfig(hidden_dim=12, message_passing_rounds=1))
    generator = torch.Generator().manual_seed(11)
    trajectory = rollout_model(model, path_problem, 4.0, generator=generator)
    assert path_problem.is_independent(trajectory.terminal_mask)
    expected = (
        model.log_partition(path_problem, 4.0)
        + trajectory.forward_log_probability
        - trajectory.backward_log_probability
        - trajectory.log_reward
    )
    assert torch.allclose(trajectory.trajectory_balance_residual, expected)
    record = trajectory.detached()
    assert record.actions[-1] == path_problem.vertex_count


def test_sampling_and_greedy_decode(path_problem: WeightedGraphProblem) -> None:
    model = GFlowNetPolicy(ModelConfig(hidden_dim=12, message_passing_rounds=1))
    samples = sample_model_masks(model, path_problem, 3.0, sample_count=12, seed=9)
    assert len(samples) == 12
    assert all(path_problem.is_independent(mask) for mask in samples)
    greedy = greedy_model_mask(model, path_problem, 3.0)
    assert path_problem.is_independent(greedy)


def test_terminal_state_has_no_actions(path_problem: WeightedGraphProblem) -> None:
    environment = IndependentSetEnvironment(path_problem)
    terminal = ConstructionState(selected_mask=0, stopped=True)
    assert not environment.valid_action_mask(terminal).any()
