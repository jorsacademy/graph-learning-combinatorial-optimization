from __future__ import annotations

import json

from gfnco.experiment import ResearchConfig, run_research_experiment, save_research_report


def test_compact_research_protocol_runs_all_shifts(tmp_path) -> None:
    config = ResearchConfig(
        training_instances=3,
        validation_instances=2,
        evaluation_instances_per_scenario=1,
        min_train_vertices=4,
        max_train_vertices=5,
        min_size_shift_vertices=6,
        max_size_shift_vertices=6,
        beta_values=(2.0,),
        evaluation_beta=2.0,
        beta_shift=4.0,
        training_steps=2,
        training_batch_size=1,
        samples_per_instance=8,
        hidden_dim=8,
        message_passing_rounds=1,
        seed=77,
    )
    gflownet, reinforce, report = run_research_experiment(config)
    assert gflownet.parameter_count > 0
    assert reinforce.parameter_count > 0
    scenarios = {row.scenario for row in report.benchmark.rows}
    assert scenarios == {
        "in_distribution",
        "size_shift",
        "sparse_graph",
        "dense_graph",
        "clustered_graph",
        "weight_shift",
        "combined_shift",
        "beta_shift",
    }
    assert all(row.feasible_rate == 1.0 for row in report.benchmark.rows)
    destination = tmp_path / "research.json"
    save_research_report(report, destination)
    payload = json.loads(destination.read_text())
    assert payload["config"]["training_instances"] == 3
