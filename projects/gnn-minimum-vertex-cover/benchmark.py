from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import networkx as nx
import torch

from vertex_cover_gnn import (
    VertexCoverGNN,
    extract_vertex_cover,
    generate_random_graph,
    networkx_to_pytorch_geometric,
    train_gnn_for_vertex_cover,
)


@dataclass(frozen=True)
class BenchmarkResult:
    seed: int
    n_nodes: int
    n_edges: int
    optimum_size: int
    gnn_size: int
    networkx_size: int
    gnn_ratio: float
    networkx_ratio: float


def is_networkx_vertex_cover(graph: nx.Graph, selected: set[int]) -> bool:
    return all(u in selected or v in selected for u, v in graph.edges())


def minimum_vertex_cover_exact(graph: nx.Graph, max_nodes: int = 30) -> set[int]:
    """Return an exact minimum vertex cover using branch-and-bound.

    This solver is intended as a benchmark oracle for small graphs, not as a
    replacement for a production MILP/CP-SAT solver. The problem is NP-hard,
    so ``max_nodes`` prevents accidental use on large instances.
    """
    if graph.is_directed():
        raise ValueError("graph must be undirected")
    if graph.number_of_nodes() > max_nodes:
        raise ValueError(f"exact solver limited to at most {max_nodes} nodes")

    self_loop_nodes = {u for u, v in nx.selfloop_edges(graph)}
    working = graph.copy()
    working.remove_edges_from(nx.selfloop_edges(working))
    working.remove_nodes_from(self_loop_nodes)

    # A maximal matching gives a fast feasible cover: both endpoints of each
    # matched edge form a valid (not necessarily minimum) vertex cover.
    matching = nx.maximal_matching(working)
    initial = set(self_loop_nodes)
    for u, v in matching:
        initial.add(u)
        initial.add(v)

    best = initial if is_networkx_vertex_cover(graph, initial) else set(graph.nodes())

    def search(remaining: nx.Graph, selected: set[int]) -> None:
        nonlocal best

        if len(selected) >= len(best):
            return
        if remaining.number_of_edges() == 0:
            best = set(selected)
            return

        # Matching size is a lower bound on how many additional vertices are
        # required. It substantially cuts the search tree on small instances.
        lower_bound = len(nx.maximal_matching(remaining))
        if len(selected) + lower_bound >= len(best):
            return

        u, v = next(iter(remaining.edges()))
        for chosen in (u, v):
            reduced = remaining.copy()
            reduced.remove_node(chosen)
            search(reduced, selected | {chosen})

    search(working, set(self_loop_nodes))
    return best


def approximation_ratio(candidate_size: int, optimum_size: int) -> float:
    if optimum_size < 0 or candidate_size < 0:
        raise ValueError("cover sizes must be non-negative")
    if optimum_size == 0:
        return 1.0 if candidate_size == 0 else float("inf")
    return candidate_size / optimum_size


def benchmark_instance(
    *,
    seed: int,
    n_nodes: int = 16,
    edge_probability: float = 0.2,
    epochs: int = 120,
    hidden_channels: int = 32,
    num_layers: int = 2,
) -> BenchmarkResult:
    """Train one GNN instance and compare it with exact and NetworkX covers."""
    torch.manual_seed(seed)
    graph = generate_random_graph(n_nodes, edge_probability, seed=seed)
    data = networkx_to_pytorch_geometric(graph)

    model = VertexCoverGNN(hidden_channels=hidden_channels, num_layers=num_layers)
    train_gnn_for_vertex_cover(model, data, num_epochs=epochs, verbose=False)
    gnn_cover = extract_vertex_cover(model, data)

    optimum = minimum_vertex_cover_exact(graph)
    nx_cover = set(nx.algorithms.approximation.min_weighted_vertex_cover(graph))

    if not is_networkx_vertex_cover(graph, gnn_cover):
        raise AssertionError("GNN post-processing returned an invalid vertex cover")
    if not is_networkx_vertex_cover(graph, nx_cover):
        raise AssertionError("NetworkX returned an invalid vertex cover")

    return BenchmarkResult(
        seed=seed,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        optimum_size=len(optimum),
        gnn_size=len(gnn_cover),
        networkx_size=len(nx_cover),
        gnn_ratio=approximation_ratio(len(gnn_cover), len(optimum)),
        networkx_ratio=approximation_ratio(len(nx_cover), len(optimum)),
    )


def run_benchmark(
    seeds: range | list[int] = range(5),
    n_nodes: int = 16,
    edge_probability: float = 0.2,
    epochs: int = 120,
) -> list[BenchmarkResult]:
    results = [
        benchmark_instance(
            seed=seed,
            n_nodes=n_nodes,
            edge_probability=edge_probability,
            epochs=epochs,
        )
        for seed in seeds
    ]

    print("seed nodes edges opt gnn nx gnn/opt nx/opt")
    for result in results:
        print(
            f"{result.seed:>4} {result.n_nodes:>5} {result.n_edges:>5} "
            f"{result.optimum_size:>3} {result.gnn_size:>3} {result.networkx_size:>2} "
            f"{result.gnn_ratio:>7.3f} {result.networkx_ratio:>6.3f}"
        )

    if results:
        print(f"Mean GNN/OPT ratio: {mean(r.gnn_ratio for r in results):.3f}")
        print(f"Mean NetworkX/OPT ratio: {mean(r.networkx_ratio for r in results):.3f}")

    return results


if __name__ == "__main__":
    run_benchmark()
