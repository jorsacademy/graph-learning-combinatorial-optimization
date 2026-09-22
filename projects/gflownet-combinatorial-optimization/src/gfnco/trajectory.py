"""Trajectory construction, sampling, and trajectory-balance accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gfnco.domain import WeightedGraphProblem
from gfnco.environment import IndependentSetEnvironment
from gfnco.model import GFlowNetPolicy


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    actions: tuple[int, ...]
    terminal_mask: int
    log_reward: float
    forward_log_probability: float
    backward_log_probability: float
    trajectory_balance_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "terminal_mask": self.terminal_mask,
            "log_reward": self.log_reward,
            "forward_log_probability": self.forward_log_probability,
            "backward_log_probability": self.backward_log_probability,
            "trajectory_balance_residual": self.trajectory_balance_residual,
        }


@dataclass(frozen=True, slots=True)
class DifferentiableTrajectory:
    actions: tuple[int, ...]
    terminal_mask: int
    log_reward: float
    forward_log_probability: Tensor
    backward_log_probability: Tensor
    entropy_sum: Tensor
    trajectory_balance_residual: Tensor

    def detached(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            actions=self.actions,
            terminal_mask=self.terminal_mask,
            log_reward=self.log_reward,
            forward_log_probability=float(self.forward_log_probability.detach().cpu()),
            backward_log_probability=float(self.backward_log_probability.detach().cpu()),
            trajectory_balance_residual=float(self.trajectory_balance_residual.detach().cpu()),
        )


def rollout_model(
    model: GFlowNetPolicy,
    problem: WeightedGraphProblem,
    beta: float,
    *,
    generator: torch.Generator | None = None,
    exploration_epsilon: float = 0.0,
    policy_temperature: float = 1.0,
) -> DifferentiableTrajectory:
    """Sample a valid trajectory and retain model log probabilities for learning."""

    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must lie in [0, 1]")
    if policy_temperature <= 0.0:
        raise ValueError("policy_temperature must be positive")
    environment = IndependentSetEnvironment(problem)
    state = environment.initial_state
    actions: list[int] = []
    forward_terms: list[Tensor] = []
    entropy_terms: list[Tensor] = []
    backward_log_probability = 0.0
    for _ in range(problem.vertex_count + 1):
        logits = model.action_logits(problem, state, beta) / policy_temperature
        log_probabilities = torch.log_softmax(logits, dim=0)
        model_probabilities = torch.softmax(logits, dim=0)
        valid_mask = torch.isfinite(logits) & (logits > torch.finfo(logits.dtype).min / 2)
        valid_count = int(torch.sum(valid_mask).item())
        if valid_count <= 0:
            raise RuntimeError("policy state has no valid actions")
        uniform = valid_mask.to(model_probabilities.dtype) / float(valid_count)
        sampling_probabilities = (
            1.0 - exploration_epsilon
        ) * model_probabilities.detach() + exploration_epsilon * uniform
        sampling_probabilities = sampling_probabilities / torch.sum(sampling_probabilities)
        action = int(
            torch.multinomial(
                sampling_probabilities,
                num_samples=1,
                generator=generator,
            ).item()
        )
        if not bool(valid_mask[action]):
            raise RuntimeError("sampler selected an invalid action")
        forward_terms.append(log_probabilities[action])
        entropy_terms.append(
            -torch.sum(model_probabilities[valid_mask] * log_probabilities[valid_mask])
        )
        actions.append(action)
        transition = environment.transition(state, action)
        backward_log_probability += transition.backward_log_probability
        state = transition.current
        if state.stopped:
            break
    if not state.stopped:
        raise RuntimeError("trajectory did not terminate within the acyclic horizon")
    terminal_mask = environment.terminal_mask(state)
    log_reward = problem.log_reward(terminal_mask, beta)
    forward_sum = torch.stack(forward_terms).sum()
    entropy_sum = torch.stack(entropy_terms).sum()
    backward_tensor = torch.tensor(
        backward_log_probability,
        dtype=forward_sum.dtype,
        device=forward_sum.device,
    )
    log_reward_tensor = torch.tensor(
        log_reward,
        dtype=forward_sum.dtype,
        device=forward_sum.device,
    )
    residual = (
        model.log_partition(problem, beta) + forward_sum - backward_tensor - log_reward_tensor
    )
    if not torch.isfinite(residual):
        raise RuntimeError("trajectory-balance residual is non-finite")
    return DifferentiableTrajectory(
        actions=tuple(actions),
        terminal_mask=terminal_mask,
        log_reward=log_reward,
        forward_log_probability=forward_sum,
        backward_log_probability=backward_tensor,
        entropy_sum=entropy_sum,
        trajectory_balance_residual=residual,
    )


def sample_model_masks(
    model: GFlowNetPolicy,
    problem: WeightedGraphProblem,
    beta: float,
    *,
    sample_count: int,
    seed: int,
    policy_temperature: float = 1.0,
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    generator = torch.Generator(device=model.device.type)
    generator.manual_seed(seed)
    masks: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(sample_count):
            trajectory = rollout_model(
                model,
                problem,
                beta,
                generator=generator,
                policy_temperature=policy_temperature,
            )
            masks.append(trajectory.terminal_mask)
    return tuple(masks)


def greedy_model_mask(
    model: GFlowNetPolicy,
    problem: WeightedGraphProblem,
    beta: float,
) -> int:
    environment = IndependentSetEnvironment(problem)
    state = environment.initial_state
    model.eval()
    with torch.no_grad():
        for _ in range(problem.vertex_count + 1):
            logits = model.action_logits(problem, state, beta)
            action = int(torch.argmax(logits).item())
            state = environment.transition(state, action).current
            if state.stopped:
                return environment.terminal_mask(state)
    raise RuntimeError("greedy model decoding failed to terminate")
