"""Trajectory-balance GFlowNet training and a mode-seeking REINFORCE control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from gfnco.domain import WeightedGraphProblem
from gfnco.model import GFlowNetPolicy
from gfnco.trajectory import rollout_model
from gfnco.utils import set_global_seed


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 1_500
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 5.0
    exploration_start: float = 0.20
    exploration_end: float = 0.02
    beta_values: tuple[float, ...] = (2.0, 4.0, 6.0)
    validation_every: int = 100
    validation_trajectories_per_problem: int = 2
    patience_checks: int = 8
    entropy_coefficient: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        positive_integers = (
            self.steps,
            self.batch_size,
            self.validation_every,
            self.validation_trajectories_per_problem,
            self.patience_checks,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("training counts must be positive")
        positive_floats = (
            self.learning_rate,
            self.gradient_clip_norm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_floats):
            raise ValueError("learning rate and gradient clip must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0.0 <= self.exploration_end <= self.exploration_start <= 1.0:
            raise ValueError("exploration schedule must lie in [0, 1] and decrease")
        if not self.beta_values or any(
            not math.isfinite(beta) or beta < 0.0 for beta in self.beta_values
        ):
            raise ValueError("beta_values must be finite and nonnegative")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    step: int
    train_metric: float
    validation_metric: float
    exploration_epsilon: float
    gradient_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    algorithm: str
    steps_completed: int
    best_validation_metric: float
    final_training_metric: float
    history: tuple[TrainingPoint, ...]
    config: TrainingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "steps_completed": self.steps_completed,
            "best_validation_metric": self.best_validation_metric,
            "final_training_metric": self.final_training_metric,
            "history": [point.to_dict() for point in self.history],
            "config": asdict(self.config),
        }


def _exploration_at(config: TrainingConfig, step: int) -> float:
    if config.steps <= 1:
        return config.exploration_end
    fraction = (step - 1) / float(config.steps - 1)
    return config.exploration_start + fraction * (config.exploration_end - config.exploration_start)


def _sample_problem_beta(
    problems: tuple[WeightedGraphProblem, ...],
    beta_values: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[WeightedGraphProblem, float]:
    problem = problems[int(rng.integers(0, len(problems)))]
    beta = beta_values[int(rng.integers(0, len(beta_values)))]
    return problem, beta


def _mean_tb_metric(
    model: GFlowNetPolicy,
    problems: tuple[WeightedGraphProblem, ...],
    config: TrainingConfig,
    *,
    seed: int,
) -> float:
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(seed)
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for problem in problems:
            for beta in config.beta_values:
                for _ in range(config.validation_trajectories_per_problem):
                    trajectory = rollout_model(
                        model,
                        problem,
                        beta,
                        generator=generator,
                    )
                    residual = float(trajectory.trajectory_balance_residual.cpu())
                    values.append(residual * residual)
    return float(np.mean(values))


def _mean_negative_reward_metric(
    model: GFlowNetPolicy,
    problems: tuple[WeightedGraphProblem, ...],
    config: TrainingConfig,
    *,
    seed: int,
) -> float:
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(seed)
    rewards: list[float] = []
    model.eval()
    with torch.no_grad():
        for problem in problems:
            for beta in config.beta_values:
                for _ in range(config.validation_trajectories_per_problem):
                    rewards.append(
                        rollout_model(
                            model,
                            problem,
                            beta,
                            generator=generator,
                        ).log_reward
                    )
    return -float(np.mean(rewards))


def _clone_state(model: GFlowNetPolicy) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_trajectory_balance(
    model: GFlowNetPolicy,
    train_problems: tuple[WeightedGraphProblem, ...],
    validation_problems: tuple[WeightedGraphProblem, ...],
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    config = config or TrainingConfig()
    if not train_problems or not validation_problems:
        raise ValueError("training and validation problem sets must be nonempty")
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = _clone_state(model)
    best_validation = float("inf")
    checks_without_improvement = 0
    history: list[TrainingPoint] = []
    final_metric = float("nan")
    steps_completed = 0

    for step in range(1, config.steps + 1):
        model.train()
        epsilon = _exploration_at(config, step)
        residuals: list[Tensor] = []
        for _ in range(config.batch_size):
            problem, beta = _sample_problem_beta(train_problems, config.beta_values, rng)
            residuals.append(
                rollout_model(
                    model,
                    problem,
                    beta,
                    generator=generator,
                    exploration_epsilon=epsilon,
                ).trajectory_balance_residual
            )
        loss = torch.mean(torch.stack(residuals).square())
        if not torch.isfinite(loss):
            raise RuntimeError("trajectory-balance loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError("trajectory-balance gradient norm became non-finite")
        optimizer.step()
        final_metric = float(loss.detach().cpu())
        steps_completed = step

        if step % config.validation_every == 0 or step == config.steps:
            validation = _mean_tb_metric(
                model,
                validation_problems,
                config,
                seed=config.seed + 20_000 + step,
            )
            history.append(
                TrainingPoint(
                    step=step,
                    train_metric=final_metric,
                    validation_metric=validation,
                    exploration_epsilon=epsilon,
                    gradient_norm=gradient_norm,
                )
            )
            if validation < best_validation - 1e-10:
                best_validation = validation
                best_state = _clone_state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= config.patience_checks:
                    break

    model.load_state_dict(best_state, strict=True)
    return TrainingSummary(
        algorithm="trajectory_balance",
        steps_completed=steps_completed,
        best_validation_metric=best_validation,
        final_training_metric=final_metric,
        history=tuple(history),
        config=config,
    )


def train_reinforce(
    model: GFlowNetPolicy,
    train_problems: tuple[WeightedGraphProblem, ...],
    validation_problems: tuple[WeightedGraphProblem, ...],
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    """Train a comparable mode-seeking policy-gradient baseline."""

    config = config or TrainingConfig()
    if not train_problems or not validation_problems:
        raise ValueError("training and validation problem sets must be nonempty")
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = _clone_state(model)
    best_validation = float("inf")
    checks_without_improvement = 0
    history: list[TrainingPoint] = []
    running_baseline = 0.0
    baseline_initialized = False
    final_metric = float("nan")
    steps_completed = 0

    for step in range(1, config.steps + 1):
        model.train()
        losses: list[Tensor] = []
        rewards: list[float] = []
        for _ in range(config.batch_size):
            problem, beta = _sample_problem_beta(train_problems, config.beta_values, rng)
            trajectory = rollout_model(model, problem, beta, generator=generator)
            rewards.append(trajectory.log_reward)
            baseline = running_baseline if baseline_initialized else trajectory.log_reward
            advantage = torch.tensor(
                trajectory.log_reward - baseline,
                dtype=trajectory.forward_log_probability.dtype,
                device=trajectory.forward_log_probability.device,
            )
            losses.append(
                -advantage.detach() * trajectory.forward_log_probability
                - config.entropy_coefficient * trajectory.entropy_sum
            )
        mean_reward = float(np.mean(rewards))
        if not baseline_initialized:
            running_baseline = mean_reward
            baseline_initialized = True
        else:
            running_baseline = 0.95 * running_baseline + 0.05 * mean_reward
        loss = torch.mean(torch.stack(losses))
        if not torch.isfinite(loss):
            raise RuntimeError("REINFORCE loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError("REINFORCE gradient norm became non-finite")
        optimizer.step()
        final_metric = -mean_reward
        steps_completed = step

        if step % config.validation_every == 0 or step == config.steps:
            validation = _mean_negative_reward_metric(
                model,
                validation_problems,
                config,
                seed=config.seed + 30_000 + step,
            )
            history.append(
                TrainingPoint(
                    step=step,
                    train_metric=final_metric,
                    validation_metric=validation,
                    exploration_epsilon=0.0,
                    gradient_norm=gradient_norm,
                )
            )
            if validation < best_validation - 1e-10:
                best_validation = validation
                best_state = _clone_state(model)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= config.patience_checks:
                    break

    model.load_state_dict(best_state, strict=True)
    return TrainingSummary(
        algorithm="reinforce",
        steps_completed=steps_completed,
        best_validation_metric=best_validation,
        final_training_metric=final_metric,
        history=tuple(history),
        config=config,
    )
