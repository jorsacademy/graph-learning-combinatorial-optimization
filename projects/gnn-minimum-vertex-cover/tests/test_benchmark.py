import networkx as nx
import pytest

from benchmark import (
    approximation_ratio,
    benchmark_instance,
    is_networkx_vertex_cover,
    minimum_vertex_cover_exact,
)


def test_exact_solver_on_empty_graph():
    graph = nx.empty_graph(6)
    cover = minimum_vertex_cover_exact(graph)
    assert cover == set()


def test_exact_solver_on_path_graph():
    graph = nx.path_graph(6)
    cover = minimum_vertex_cover_exact(graph)
    assert is_networkx_vertex_cover(graph, cover)
    assert len(cover) == 3


def test_exact_solver_on_cycle_graph():
    graph = nx.cycle_graph(7)
    cover = minimum_vertex_cover_exact(graph)
    assert is_networkx_vertex_cover(graph, cover)
    assert len(cover) == 4


def test_exact_solver_on_complete_graph():
    graph = nx.complete_graph(6)
    cover = minimum_vertex_cover_exact(graph)
    assert is_networkx_vertex_cover(graph, cover)
    assert len(cover) == 5


def test_exact_solver_handles_self_loop():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 0)
    graph.add_edge(1, 2)
    cover = minimum_vertex_cover_exact(graph)
    assert 0 in cover
    assert is_networkx_vertex_cover(graph, cover)
    assert len(cover) == 2


def test_exact_solver_rejects_large_instances():
    with pytest.raises(ValueError, match="limited"):
        minimum_vertex_cover_exact(nx.empty_graph(31), max_nodes=30)


def test_approximation_ratio_handles_zero_optimum():
    assert approximation_ratio(0, 0) == 1.0
    assert approximation_ratio(1, 0) == float("inf")


def test_benchmark_smoke_test_returns_valid_metrics():
    result = benchmark_instance(
        seed=3,
        n_nodes=8,
        edge_probability=0.25,
        epochs=2,
        hidden_channels=8,
        num_layers=1,
    )
    assert result.optimum_size <= result.gnn_size
    assert result.optimum_size <= result.networkx_size
    assert result.gnn_ratio >= 1.0
    assert result.networkx_ratio >= 1.0
