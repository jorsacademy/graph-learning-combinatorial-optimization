import networkx as nx
import pytest
import torch

from vertex_cover_gnn import (
    VertexCoverGNN,
    generate_random_graph,
    is_vertex_cover,
    networkx_to_pytorch_geometric,
    repair_and_prune_vertex_cover,
    train_gnn_for_vertex_cover,
    unique_undirected_edges,
    vertex_cover_loss,
)


def test_generate_random_graph_is_reproducible():
    g1 = generate_random_graph(20, 0.2, seed=123)
    g2 = generate_random_graph(20, 0.2, seed=123)
    assert sorted(g1.edges()) == sorted(g2.edges())


def test_empty_graph_conversion_has_valid_edge_index_shape():
    graph = nx.empty_graph(4)
    data = networkx_to_pytorch_geometric(graph)

    assert data.edge_index.shape == (2, 0)
    assert data.x.shape == (4, 1)
    assert torch.equal(data.x, torch.zeros((4, 1)))


def test_single_node_forward_has_stable_shape_and_probability_range():
    graph = nx.empty_graph(1)
    data = networkx_to_pytorch_geometric(graph)
    model = VertexCoverGNN(hidden_channels=8, num_layers=1)

    output = model(data.x, data.edge_index)

    assert output.shape == (1, 1)
    assert 0.0 <= output.item() <= 1.0


def test_non_consecutive_node_labels_are_rejected():
    graph = nx.Graph()
    graph.add_edge(10, 20)

    with pytest.raises(ValueError, match="consecutive integers"):
        networkx_to_pytorch_geometric(graph)


def test_unique_undirected_edges_removes_reverse_duplicates():
    edge_index = torch.tensor(
        [[0, 1, 1, 0, 1, 2], [1, 0, 0, 1, 2, 1]], dtype=torch.long
    )

    unique = unique_undirected_edges(edge_index)
    pairs = {tuple(pair) for pair in unique.t().tolist()}

    assert pairs == {(0, 1), (1, 2)}


def test_vertex_cover_loss_is_finite_with_no_edges():
    scores = torch.tensor([0.2, 0.7], requires_grad=True)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    losses = vertex_cover_loss(scores, edge_index)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert losses.coverage.item() == 0.0
    assert scores.grad is not None


def test_repair_returns_valid_cover_when_all_scores_are_below_threshold():
    graph = nx.cycle_graph(7)
    data = networkx_to_pytorch_geometric(graph)
    scores = torch.linspace(0.1, 0.4, steps=7)

    cover = repair_and_prune_vertex_cover(scores, data.edge_index, threshold=0.5)

    assert is_vertex_cover(data.edge_index, cover)


def test_triangle_repairs_to_two_vertices():
    graph = nx.complete_graph(3)
    data = networkx_to_pytorch_geometric(graph)
    scores = torch.tensor([0.1, 0.2, 0.3])

    cover = repair_and_prune_vertex_cover(scores, data.edge_index, threshold=0.5)

    assert is_vertex_cover(data.edge_index, cover)
    assert len(cover) == 2


def test_empty_graph_repairs_to_empty_cover():
    graph = nx.empty_graph(5)
    data = networkx_to_pytorch_geometric(graph)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.49])

    cover = repair_and_prune_vertex_cover(scores, data.edge_index, threshold=0.5)

    assert cover == set()
    assert is_vertex_cover(data.edge_index, cover)


def test_repair_is_valid_across_many_random_graphs():
    generator = torch.Generator().manual_seed(77)

    for seed in range(50):
        graph = generate_random_graph(20, 0.2, seed=seed)
        data = networkx_to_pytorch_geometric(graph)
        scores = torch.rand(20, generator=generator)
        cover = repair_and_prune_vertex_cover(scores, data.edge_index, threshold=0.5)
        assert is_vertex_cover(data.edge_index, cover)


def test_short_training_smoke_test_runs_and_parameters_remain_finite():
    torch.manual_seed(42)
    graph = nx.path_graph(6)
    data = networkx_to_pytorch_geometric(graph)
    model = VertexCoverGNN(hidden_channels=8, num_layers=2)

    train_gnn_for_vertex_cover(model, data, num_epochs=3, lr=1e-3, verbose=False)

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
