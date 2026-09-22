"""Deterministic runtime helpers."""

from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.use_deterministic_algorithms(True)


def write_json(payload: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
