from __future__ import annotations

import json

from gfnco.cli import main


def test_cli_generate_oracle_collect_train_sample_and_benchmark(tmp_path) -> None:
    problem = tmp_path / "problem.json"
    assert (
        main(
            [
                "generate",
                "--vertices",
                "5",
                "--seed",
                "4",
                "--output",
                str(problem),
            ]
        )
        == 0
    )
    oracle = tmp_path / "oracle.json"
    assert (
        main(
            [
                "oracle",
                "--input",
                str(problem),
                "--beta",
                "4",
                "--output",
                str(oracle),
            ]
        )
        == 0
    )
    assert json.loads(oracle.read_text())["exact"]["independent_set_count"] > 0

    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    assert (
        main(
            [
                "collect",
                "--instances",
                "3",
                "--min-vertices",
                "4",
                "--max-vertices",
                "5",
                "--seed",
                "10",
                "--output",
                str(train),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--instances",
                "2",
                "--min-vertices",
                "4",
                "--max-vertices",
                "5",
                "--seed",
                "20",
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    checkpoint = tmp_path / "gfn.safetensors"
    training_report = tmp_path / "train.json"
    assert (
        main(
            [
                "train",
                str(train),
                "--validation",
                str(validation),
                "--algorithm",
                "trajectory_balance",
                "--steps",
                "2",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--rounds",
                "1",
                "--checkpoint",
                str(checkpoint),
                "--output-report",
                str(training_report),
            ]
        )
        == 0
    )
    assert checkpoint.exists()
    samples = tmp_path / "samples.json"
    assert (
        main(
            [
                "sample",
                "--input",
                str(problem),
                "--checkpoint",
                str(checkpoint),
                "--samples",
                "5",
                "--output",
                str(samples),
            ]
        )
        == 0
    )
    assert len(json.loads(samples.read_text())["samples"]) == 5
    benchmark = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"
    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--gflownet-checkpoint",
                str(checkpoint),
                "--samples",
                "10",
                "--output-json",
                str(benchmark),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )
    assert benchmark.exists()
    assert benchmark_csv.exists()


def test_cli_returns_structured_error(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert main(["oracle", "--input", str(missing)]) == 1
