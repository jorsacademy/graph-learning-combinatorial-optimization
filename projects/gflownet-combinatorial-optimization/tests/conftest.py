from __future__ import annotations

import pytest

from gfnco.domain import WeightedGraphProblem


@pytest.fixture
def path_problem() -> WeightedGraphProblem:
    return WeightedGraphProblem(
        name="path-3",
        weights=(1.0, 2.0, 3.0),
        edges=((0, 1), (1, 2)),
        seed=7,
    )
