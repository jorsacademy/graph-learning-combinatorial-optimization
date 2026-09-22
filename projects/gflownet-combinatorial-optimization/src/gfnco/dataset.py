"""Versioned graph corpora for GFlowNet training and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from gfnco.domain import WeightedGraphProblem
from gfnco.generator import GraphRegime, generate_problems

CORPUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ProblemCorpus:
    problems: tuple[WeightedGraphProblem, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.problems:
            raise ValueError("corpus must contain at least one problem")
        names = [problem.name for problem in self.problems]
        if len(names) != len(set(names)):
            raise ValueError("problem names must be unique")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            [problem.to_dict() for problem in self.problems],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({problem.regime for problem in self.problems}))

    def to_summary(self) -> dict[str, object]:
        return {
            "problem_count": len(self.problems),
            "min_vertices": min(problem.vertex_count for problem in self.problems),
            "max_vertices": max(problem.vertex_count for problem in self.problems),
            "regimes": list(self.regimes),
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }


def collect_corpus(
    *,
    count: int,
    min_vertices: int,
    max_vertices: int,
    seed: int,
    regimes: tuple[GraphRegime, ...] = ("in_distribution",),
    edge_probability: float = 0.30,
) -> ProblemCorpus:
    problems = generate_problems(
        count=count,
        min_vertices=min_vertices,
        max_vertices=max_vertices,
        seed=seed,
        regimes=regimes,
        edge_probability=edge_probability,
    )
    return ProblemCorpus(
        problems=problems,
        metadata={
            "generator": "gfnco",
            "count": count,
            "min_vertices": min_vertices,
            "max_vertices": max_vertices,
            "seed": seed,
            "regimes": list(regimes),
            "edge_probability": edge_probability,
        },
    )


def save_corpus(corpus: ProblemCorpus, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "manifest",
        "schema_version": CORPUS_SCHEMA_VERSION,
        "problem_count": len(corpus.problems),
        "fingerprint": corpus.fingerprint,
        "metadata": corpus.metadata,
    }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
        for problem in corpus.problems:
            handle.write(
                json.dumps(
                    {"type": "problem", "problem": problem.to_dict()},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_corpus(path: str | Path) -> ProblemCorpus:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("corpus file is empty")
    manifest = json.loads(lines[0])
    if not isinstance(manifest, dict) or manifest.get("type") != "manifest":
        raise ValueError("first corpus line must be a manifest")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus schema version")
    problems: list[WeightedGraphProblem] = []
    for line_number, line in enumerate(lines[1:], start=2):
        payload = json.loads(line)
        if not isinstance(payload, dict) or payload.get("type") != "problem":
            raise ValueError(f"line {line_number} is not a problem record")
        problem_payload = cast(dict[str, object], payload["problem"])
        problems.append(WeightedGraphProblem.from_dict(problem_payload))
    corpus = ProblemCorpus(
        problems=tuple(problems),
        metadata=cast(dict[str, object], manifest.get("metadata", {})),
    )
    if len(corpus.problems) != int(manifest["problem_count"]):
        raise ValueError("problem count does not match the manifest")
    if corpus.fingerprint != str(manifest["fingerprint"]):
        raise ValueError("corpus fingerprint does not match the manifest")
    return corpus


def split_corpus(
    corpus: ProblemCorpus,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[ProblemCorpus, ProblemCorpus]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if len(corpus.problems) < 2:
        raise ValueError("at least two problems are required for a split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus.problems))
    validation_count = max(
        1,
        min(len(corpus.problems) - 1, round(validation_fraction * len(corpus.problems))),
    )
    validation_indices = {int(index) for index in order[:validation_count]}
    train = tuple(
        problem for index, problem in enumerate(corpus.problems) if index not in validation_indices
    )
    validation = tuple(
        problem for index, problem in enumerate(corpus.problems) if index in validation_indices
    )
    return (
        ProblemCorpus(train, {**corpus.metadata, "split": "train", "split_seed": seed}),
        ProblemCorpus(
            validation,
            {**corpus.metadata, "split": "validation", "split_seed": seed},
        ),
    )
