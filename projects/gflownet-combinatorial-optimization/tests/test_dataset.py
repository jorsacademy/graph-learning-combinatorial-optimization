from __future__ import annotations

import json

import pytest

from gfnco.dataset import collect_corpus, load_corpus, save_corpus, split_corpus


def test_corpus_round_trip_and_split(tmp_path) -> None:
    corpus = collect_corpus(
        count=6,
        min_vertices=5,
        max_vertices=7,
        seed=10,
        regimes=("in_distribution", "sparse"),
    )
    path = tmp_path / "corpus.jsonl"
    save_corpus(corpus, path)
    loaded = load_corpus(path)
    assert loaded.fingerprint == corpus.fingerprint
    assert loaded.regimes == ("in_distribution", "sparse")
    train, validation = split_corpus(loaded, validation_fraction=0.34, seed=3)
    assert len(train.problems) + len(validation.problems) == 6
    assert {p.name for p in train.problems}.isdisjoint({p.name for p in validation.problems})


def test_corpus_detects_tampering(tmp_path) -> None:
    corpus = collect_corpus(count=2, min_vertices=4, max_vertices=4, seed=2)
    path = tmp_path / "corpus.jsonl"
    save_corpus(corpus, path)
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["problem"]["weights"][0] += 1.0
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_corpus(path)
