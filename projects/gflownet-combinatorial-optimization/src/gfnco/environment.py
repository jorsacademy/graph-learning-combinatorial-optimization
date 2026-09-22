"""A constructive acyclic MDP for weighted independent sets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gfnco.domain import WeightedGraphProblem


@dataclass(frozen=True, slots=True)
class ConstructionState:
    selected_mask: int = 0
    stopped: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    previous: ConstructionState
    action: int
    current: ConstructionState
    backward_log_probability: float


class IndependentSetEnvironment:
    """Add feasible vertices in any order, then emit a dedicated stop action."""

    def __init__(self, problem: WeightedGraphProblem) -> None:
        self.problem = problem
        self.stop_action = problem.vertex_count

    @property
    def initial_state(self) -> ConstructionState:
        return ConstructionState()

    def valid_action_mask(self, state: ConstructionState) -> np.ndarray:
        mask = np.zeros(self.problem.vertex_count + 1, dtype=bool)
        if state.stopped:
            return mask
        available = self.problem.available_mask(state.selected_mask)
        for vertex in range(self.problem.vertex_count):
            mask[vertex] = bool(available & (1 << vertex))
        mask[self.stop_action] = True
        return mask

    def valid_actions(self, state: ConstructionState) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.valid_action_mask(state)))

    def transition(self, state: ConstructionState, action: int) -> Transition:
        if state.stopped:
            raise ValueError("terminal states have no outgoing actions")
        if action == self.stop_action:
            current = ConstructionState(selected_mask=state.selected_mask, stopped=True)
            return Transition(
                previous=state,
                action=action,
                current=current,
                backward_log_probability=0.0,
            )
        if not 0 <= action < self.problem.vertex_count:
            raise ValueError("action is outside the valid range")
        next_mask = self.problem.add_vertex(state.selected_mask, action)
        selected_count = next_mask.bit_count()
        if selected_count <= 0:
            raise RuntimeError("an add transition must create a nonempty state")
        current = ConstructionState(selected_mask=next_mask, stopped=False)
        return Transition(
            previous=state,
            action=action,
            current=current,
            backward_log_probability=-math.log(float(selected_count)),
        )

    def terminal_mask(self, state: ConstructionState) -> int:
        if not state.stopped:
            raise ValueError("state is not terminal")
        if not self.problem.is_independent(state.selected_mask):
            raise RuntimeError("environment reached an infeasible terminal state")
        return state.selected_mask
