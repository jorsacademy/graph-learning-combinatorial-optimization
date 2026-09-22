"""Frozen training, baseline, and distribution-shift research protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gfnco.dataset import ProblemCorpus, collect_corpus
from gfnco.evaluation import (
    BenchmarkReport,
    SamplingMetrics,
    build_benchmark_report,
    evaluate_problem,
)
from gfnco.generator import GraphRegime
from gfnco.model import GFlowNetPolicy, ModelConfig
from gfnco.training import (
    TrainingConfig,
    TrainingSummary,
    train_reinforce,
    train_trajectory_balance,
)
from gfnco.utils import set_global_seed


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    training_instances: int = 48
    validation_instances: int = 12
    evaluation_instances_per_scenario: int = 6
    min_train_vertices: int = 8
    max_train_vertices: int = 12
    min_size_shift_vertices: int = 14
    max_size_shift_vertices: int = 16
    edge_probability: float = 0.30
    beta_values: tuple[float, ...] = (2.0, 4.0, 6.0)
    evaluation_beta: float = 6.0
    beta_shift: float = 10.0
    training_steps: int = 1_500
    training_batch_size: int = 8
    samples_per_instance: int = 512
    hidden_dim: int = 64
    message_passing_rounds: int = 3
    seed: int = 2026

    def __post_init__(self) -> None:
        counts = (
            self.training_instances,
            self.validation_instances,
            self.evaluation_instances_per_scenario,
            self.min_train_vertices,
            self.max_train_vertices,
            self.min_size_shift_vertices,
            self.max_size_shift_vertices,
            self.training_steps,
            self.training_batch_size,
            self.samples_per_instance,
            self.hidden_dim,
            self.message_passing_rounds,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("research counts and dimensions must be positive")
        if self.max_train_vertices < self.min_train_vertices:
            raise ValueError("invalid training vertex range")
        if self.max_size_shift_vertices < self.min_size_shift_vertices:
            raise ValueError("invalid size-shift vertex range")
        if self.min_size_shift_vertices <= self.max_train_vertices:
            raise ValueError("size-shift vertices must exceed the training range")
        if not 0.0 <= self.edge_probability <= 1.0:
            raise ValueError("edge_probability must lie in [0, 1]")
        if not self.beta_values or any(beta < 0.0 for beta in self.beta_values):
            raise ValueError("beta_values must be nonempty and nonnegative")
        if self.evaluation_beta < 0.0 or self.beta_shift < 0.0:
            raise ValueError("evaluation beta values must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: ResearchConfig
    training_corpus: dict[str, object]
    validation_corpus: dict[str, object]
    gflownet_training: TrainingSummary
    reinforce_training: TrainingSummary
    benchmark: BenchmarkReport

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "training_corpus": self.training_corpus,
            "validation_corpus": self.validation_corpus,
            "gflownet_training": self.gflownet_training.to_dict(),
            "reinforce_training": self.reinforce_training.to_dict(),
            "benchmark": self.benchmark.to_dict(),
        }


def _training_config(config: ResearchConfig, *, seed: int) -> TrainingConfig:
    return TrainingConfig(
        steps=config.training_steps,
        batch_size=config.training_batch_size,
        beta_values=config.beta_values,
        validation_every=max(1, config.training_steps // 10),
        validation_trajectories_per_problem=1,
        patience_checks=6,
        seed=seed,
    )


def _evaluation_corpus(
    config: ResearchConfig,
    *,
    regime: GraphRegime,
    seed: int,
    size_shift: bool = False,
) -> ProblemCorpus:
    min_vertices = config.min_size_shift_vertices if size_shift else config.min_train_vertices
    max_vertices = config.max_size_shift_vertices if size_shift else config.max_train_vertices
    return collect_corpus(
        count=config.evaluation_instances_per_scenario,
        min_vertices=min_vertices,
        max_vertices=max_vertices,
        seed=seed,
        regimes=(regime,),
        edge_probability=config.edge_probability,
    )


def run_research_experiment(
    config: ResearchConfig | None = None,
) -> tuple[GFlowNetPolicy, GFlowNetPolicy, ResearchReport]:
    config = config or ResearchConfig()
    training_corpus = collect_corpus(
        count=config.training_instances,
        min_vertices=config.min_train_vertices,
        max_vertices=config.max_train_vertices,
        seed=config.seed + 1_000,
        regimes=("in_distribution",),
        edge_probability=config.edge_probability,
    )
    validation_corpus = collect_corpus(
        count=config.validation_instances,
        min_vertices=config.min_train_vertices,
        max_vertices=config.max_train_vertices,
        seed=config.seed + 2_000,
        regimes=("in_distribution",),
        edge_probability=config.edge_probability,
    )
    model_config = ModelConfig(
        hidden_dim=config.hidden_dim,
        message_passing_rounds=config.message_passing_rounds,
        beta_scale=max(10.0, config.beta_shift),
    )

    set_global_seed(config.seed + 3_000)
    gflownet = GFlowNetPolicy(model_config)
    gflownet_summary = train_trajectory_balance(
        gflownet,
        training_corpus.problems,
        validation_corpus.problems,
        config=_training_config(config, seed=config.seed + 3_000),
    )

    set_global_seed(config.seed + 4_000)
    reinforce = GFlowNetPolicy(model_config)
    reinforce_summary = train_reinforce(
        reinforce,
        training_corpus.problems,
        validation_corpus.problems,
        config=_training_config(config, seed=config.seed + 4_000),
    )

    scenario_specs: tuple[tuple[str, GraphRegime, int, bool, float], ...] = (
        ("in_distribution", "in_distribution", config.seed + 10_000, False, config.evaluation_beta),
        ("size_shift", "in_distribution", config.seed + 20_000, True, config.evaluation_beta),
        ("sparse_graph", "sparse", config.seed + 30_000, False, config.evaluation_beta),
        ("dense_graph", "dense", config.seed + 40_000, False, config.evaluation_beta),
        ("clustered_graph", "clustered", config.seed + 50_000, False, config.evaluation_beta),
        ("weight_shift", "weight_lognormal", config.seed + 60_000, False, config.evaluation_beta),
        ("combined_shift", "combined_shift", config.seed + 70_000, False, config.evaluation_beta),
        ("beta_shift", "in_distribution", config.seed + 80_000, False, config.beta_shift),
    )
    rows: list[SamplingMetrics] = []
    scenario_fingerprints: dict[str, str] = {}
    for scenario, regime, scenario_seed, size_shift, beta in scenario_specs:
        corpus = _evaluation_corpus(
            config,
            regime=regime,
            seed=scenario_seed,
            size_shift=size_shift,
        )
        scenario_fingerprints[scenario] = corpus.fingerprint
        for problem_index, problem in enumerate(corpus.problems):
            rows.extend(
                evaluate_problem(
                    problem,
                    beta=beta,
                    sample_count=config.samples_per_instance,
                    seed=scenario_seed + problem_index * 101,
                    gflownet_model=gflownet,
                    reinforce_model=reinforce,
                    scenario=scenario,
                )
            )
    benchmark = build_benchmark_report(
        tuple(rows),
        metadata={
            "training_corpus_fingerprint": training_corpus.fingerprint,
            "validation_corpus_fingerprint": validation_corpus.fingerprint,
            "scenario_fingerprints": scenario_fingerprints,
            "target": "p(x) proportional to exp(beta * weight(x) / total_vertex_weight)",
            "exact_support": "complete enumeration for every reported problem",
        },
    )
    report = ResearchReport(
        config=config,
        training_corpus=training_corpus.to_summary(),
        validation_corpus=validation_corpus.to_summary(),
        gflownet_training=gflownet_summary,
        reinforce_training=reinforce_summary,
        benchmark=benchmark,
    )
    return gflownet, reinforce, report


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
