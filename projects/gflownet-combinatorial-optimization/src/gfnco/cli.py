"""Command-line interface for GFlowNet combinatorial-optimization experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from gfnco.dataset import collect_corpus, load_corpus, save_corpus
from gfnco.evaluation import (
    SamplingMetrics,
    build_benchmark_report,
    evaluate_problem,
    save_report_csv,
    save_report_json,
)
from gfnco.experiment import ResearchConfig, run_research_experiment, save_research_report
from gfnco.features import feature_schema
from gfnco.generator import SUPPORTED_REGIMES, GeneratorConfig, GraphRegime, generate_problem
from gfnco.model import GFlowNetPolicy, ModelConfig, load_checkpoint, save_checkpoint
from gfnco.oracle import build_target_distribution, enumerate_independent_sets
from gfnco.training import TrainingConfig, train_reinforce, train_trajectory_balance
from gfnco.trajectory import sample_model_masks
from gfnco.utils import write_json


def _write_or_print(payload: dict[str, object], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        write_json(payload, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfnco",
        description="Trajectory-balance GFlowNet benchmark for weighted independent sets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema", help="print the feature and action schema")
    schema.add_argument("--output", type=Path)

    generate = subparsers.add_parser("generate", help="generate one deterministic MWIS instance")
    generate.add_argument("--vertices", type=int, default=12)
    generate.add_argument("--edge-probability", type=float, default=0.30)
    generate.add_argument("--regime", choices=SUPPORTED_REGIMES, default="in_distribution")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="create a versioned graph corpus")
    collect.add_argument("--instances", type=int, default=32)
    collect.add_argument("--min-vertices", type=int, default=8)
    collect.add_argument("--max-vertices", type=int, default=12)
    collect.add_argument("--edge-probability", type=float, default=0.30)
    collect.add_argument(
        "--regimes",
        nargs="+",
        choices=SUPPORTED_REGIMES,
        default=["in_distribution"],
    )
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--output", type=Path, required=True)

    oracle = subparsers.add_parser("oracle", help="enumerate exact support and target statistics")
    oracle.add_argument("--input", type=Path, required=True)
    oracle.add_argument("--beta", type=float, default=6.0)
    oracle.add_argument("--output", type=Path)

    train = subparsers.add_parser("train", help="train trajectory balance or REINFORCE")
    train.add_argument("train_corpus", type=Path)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--algorithm", choices=("trajectory_balance", "reinforce"), required=True)
    train.add_argument("--steps", type=int, default=1_500)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--rounds", type=int, default=3)
    train.add_argument("--betas", nargs="+", type=float, default=[2.0, 4.0, 6.0])
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output-report", type=Path)

    sample = subparsers.add_parser(
        "sample",
        help="sample feasible independent sets from a checkpoint",
    )
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--beta", type=float, default=6.0)
    sample.add_argument("--samples", type=int, default=64)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--output", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="compare samplers against exact target laws",
    )
    benchmark.add_argument("corpus", type=Path)
    benchmark.add_argument("--gflownet-checkpoint", type=Path)
    benchmark.add_argument("--reinforce-checkpoint", type=Path)
    benchmark.add_argument("--beta", type=float, default=6.0)
    benchmark.add_argument("--samples", type=int, default=512)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--scenario", default="benchmark")
    benchmark.add_argument("--output-json", type=Path)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser("research", help="run the frozen train-and-shift protocol")
    research.add_argument("--training-instances", type=int, default=48)
    research.add_argument("--validation-instances", type=int, default=12)
    research.add_argument("--evaluation-instances", type=int, default=6)
    research.add_argument("--training-steps", type=int, default=1_500)
    research.add_argument("--training-batch-size", type=int, default=8)
    research.add_argument("--samples", type=int, default=512)
    research.add_argument("--hidden-dim", type=int, default=64)
    research.add_argument("--rounds", type=int, default=3)
    research.add_argument("--seed", type=int, default=2026)
    research.add_argument("--gflownet-checkpoint", type=Path, required=True)
    research.add_argument("--reinforce-checkpoint", type=Path, required=True)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_values=tuple(float(beta) for beta in args.betas),
        validation_every=max(1, args.steps // 10),
        validation_trajectories_per_problem=1,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            _write_or_print(feature_schema(), args.output)
            return 0

        if args.command == "generate":
            problem = generate_problem(
                GeneratorConfig(
                    vertex_count=args.vertices,
                    edge_probability=args.edge_probability,
                    regime=cast(GraphRegime, args.regime),
                    seed=args.seed,
                )
            )
            from gfnco.domain import save_problem

            save_problem(problem, args.output)
            _write_or_print(
                {"output": str(args.output), "problem": problem.to_dict()},
                None,
            )
            return 0

        if args.command == "collect":
            regimes = tuple(cast(GraphRegime, regime) for regime in args.regimes)
            corpus = collect_corpus(
                count=args.instances,
                min_vertices=args.min_vertices,
                max_vertices=args.max_vertices,
                seed=args.seed,
                regimes=regimes,
                edge_probability=args.edge_probability,
            )
            save_corpus(corpus, args.output)
            _write_or_print({"output": str(args.output), **corpus.to_summary()}, None)
            return 0

        if args.command == "oracle":
            from gfnco.domain import load_problem

            problem = load_problem(args.input)
            exact = enumerate_independent_sets(problem)
            target = build_target_distribution(problem, exact, args.beta)
            _write_or_print(
                {
                    "problem": problem.name,
                    "exact": exact.to_dict(),
                    "target": target.to_dict(),
                },
                args.output,
            )
            return 0

        if args.command == "train":
            train_corpus = load_corpus(args.train_corpus)
            validation_corpus = load_corpus(args.validation)
            model = GFlowNetPolicy(
                ModelConfig(
                    hidden_dim=args.hidden_dim,
                    message_passing_rounds=args.rounds,
                    beta_scale=max(10.0, max(args.betas)),
                )
            )
            training_config = _training_config(args)
            summary = (
                train_trajectory_balance(
                    model,
                    train_corpus.problems,
                    validation_corpus.problems,
                    config=training_config,
                )
                if args.algorithm == "trajectory_balance"
                else train_reinforce(
                    model,
                    train_corpus.problems,
                    validation_corpus.problems,
                    config=training_config,
                )
            )
            save_checkpoint(
                model,
                args.checkpoint,
                metadata={
                    "algorithm": args.algorithm,
                    "train_corpus_fingerprint": train_corpus.fingerprint,
                    "validation_corpus_fingerprint": validation_corpus.fingerprint,
                    "training_summary": summary.to_dict(),
                },
            )
            payload = {"checkpoint": str(args.checkpoint), **summary.to_dict()}
            _write_or_print(payload, args.output_report)
            return 0

        if args.command == "sample":
            from gfnco.domain import load_problem

            problem = load_problem(args.input)
            model, metadata = load_checkpoint(args.checkpoint)
            masks = sample_model_masks(
                model,
                problem,
                args.beta,
                sample_count=args.samples,
                seed=args.seed,
            )
            _write_or_print(
                {
                    "problem": problem.name,
                    "beta": args.beta,
                    "checkpoint_metadata": metadata,
                    "samples": [
                        {
                            "mask": mask,
                            "decision": list(problem.mask_to_decision(mask)),
                            "objective": problem.objective(mask),
                        }
                        for mask in masks
                    ],
                },
                args.output,
            )
            return 0

        if args.command == "benchmark":
            corpus = load_corpus(args.corpus)
            gflownet = (
                load_checkpoint(args.gflownet_checkpoint)[0] if args.gflownet_checkpoint else None
            )
            reinforce = (
                load_checkpoint(args.reinforce_checkpoint)[0] if args.reinforce_checkpoint else None
            )
            rows: list[SamplingMetrics] = []
            for index, problem in enumerate(corpus.problems):
                rows.extend(
                    evaluate_problem(
                        problem,
                        beta=args.beta,
                        sample_count=args.samples,
                        seed=args.seed + 101 * index,
                        gflownet_model=gflownet,
                        reinforce_model=reinforce,
                        scenario=args.scenario,
                    )
                )
            benchmark_report = build_benchmark_report(
                tuple(rows),
                metadata={
                    "corpus_fingerprint": corpus.fingerprint,
                    "beta": args.beta,
                    "sample_count": args.samples,
                },
            )
            if args.output_json:
                save_report_json(benchmark_report, args.output_json)
            if args.output_csv:
                save_report_csv(benchmark_report, args.output_csv)
            _write_or_print(benchmark_report.to_dict(), None)
            return 0

        research_config = ResearchConfig(
            training_instances=args.training_instances,
            validation_instances=args.validation_instances,
            evaluation_instances_per_scenario=args.evaluation_instances,
            training_steps=args.training_steps,
            training_batch_size=args.training_batch_size,
            samples_per_instance=args.samples,
            hidden_dim=args.hidden_dim,
            message_passing_rounds=args.rounds,
            seed=args.seed,
        )
        gflownet, reinforce, research_report = run_research_experiment(research_config)
        save_checkpoint(
            gflownet,
            args.gflownet_checkpoint,
            metadata={
                "algorithm": "trajectory_balance",
                "research_config": research_report.to_dict()["config"],
                "training_corpus": research_report.training_corpus,
            },
        )
        save_checkpoint(
            reinforce,
            args.reinforce_checkpoint,
            metadata={
                "algorithm": "reinforce",
                "research_config": research_report.to_dict()["config"],
                "training_corpus": research_report.training_corpus,
            },
        )
        save_research_report(research_report, args.output_report)
        _write_or_print(
            {
                "gflownet_checkpoint": str(args.gflownet_checkpoint),
                "reinforce_checkpoint": str(args.reinforce_checkpoint),
                "report": str(args.output_report),
            },
            None,
        )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
