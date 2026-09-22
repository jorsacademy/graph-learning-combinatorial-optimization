from __future__ import annotations

import math

import torch

from gfnco.generator import generate_problems
from gfnco.model import GFlowNetPolicy, ModelConfig
from gfnco.training import TrainingConfig, train_reinforce, train_trajectory_balance


def _tiny_problem_sets():
    train = generate_problems(count=4, min_vertices=5, max_vertices=6, seed=21)
    validation = generate_problems(count=2, min_vertices=5, max_vertices=6, seed=31)
    return train, validation


def test_trajectory_balance_training_updates_model() -> None:
    train, validation = _tiny_problem_sets()
    torch.manual_seed(7)
    model = GFlowNetPolicy(ModelConfig(hidden_dim=12, message_passing_rounds=1))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    summary = train_trajectory_balance(
        model,
        train,
        validation,
        config=TrainingConfig(
            steps=4,
            batch_size=2,
            beta_values=(2.0, 4.0),
            validation_every=2,
            validation_trajectories_per_problem=1,
            patience_checks=3,
            seed=8,
        ),
    )
    assert summary.algorithm == "trajectory_balance"
    assert summary.steps_completed == 4
    assert math.isfinite(summary.best_validation_metric)
    assert len(summary.history) == 2
    assert any(not torch.allclose(before[key], model.state_dict()[key]) for key in before)


def test_reinforce_training_runs_as_mode_seeking_control() -> None:
    train, validation = _tiny_problem_sets()
    model = GFlowNetPolicy(ModelConfig(hidden_dim=12, message_passing_rounds=1))
    summary = train_reinforce(
        model,
        train,
        validation,
        config=TrainingConfig(
            steps=4,
            batch_size=2,
            beta_values=(3.0,),
            validation_every=2,
            validation_trajectories_per_problem=1,
            patience_checks=3,
            entropy_coefficient=0.01,
            seed=9,
        ),
    )
    assert summary.algorithm == "reinforce"
    assert summary.steps_completed == 4
    assert math.isfinite(summary.best_validation_metric)
    assert summary.to_dict()["config"]["steps"] == 4
