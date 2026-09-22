from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


class VertexCoverGNN(nn.Module):
    """GCN that predicts a soft probability of selecting each node."""

    def __init__(self, hidden_channels: int = 64, num_layers: int = 3):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_encoder = nn.Linear(1, hidden_channels)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.node_encoder(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return torch.sigmoid(self.output(x))


def generate_random_graph(
    n_nodes: int, edge_probability: float = 0.3, seed: int | None = None
) -> nx.Graph:
    """Generate a reproducible Erdős–Rényi graph."""
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be in [0, 1]")
    return nx.erdos_renyi_graph(n_nodes, edge_probability, seed=seed)


def networkx_to_pytorch_geometric(graph: nx.Graph) -> Data:
    """Convert an undirected NetworkX graph to a PyTorch Geometric Data object."""
    n_nodes = graph.number_of_nodes()

    # This project assumes integer node IDs 0..n-1 so tensor indexing is unambiguous.
    expected_nodes = list(range(n_nodes))
    if sorted(graph.nodes()) != expected_nodes:
        raise ValueError("Graph nodes must be labeled with consecutive integers 0..n-1")

    edges = list(graph.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=n_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    degrees = torch.tensor(
        [graph.degree(node) for node in range(n_nodes)], dtype=torch.float32
    ).view(-1, 1)
    max_degree = degrees.max().item() if n_nodes else 0.0
    if max_degree > 0:
        degrees = degrees / max_degree

    return Data(x=degrees, edge_index=edge_index, num_nodes=n_nodes)


def unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Return each undirected edge once as shape [2, E_unique]."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))

    u, v = edge_index
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    pairs = torch.stack((lo, hi), dim=1)
    pairs = torch.unique(pairs, dim=0)
    return pairs.t().contiguous()


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    size: torch.Tensor
    coverage: torch.Tensor


def vertex_cover_loss(
    node_scores: torch.Tensor,
    edge_index: torch.Tensor,
    coverage_target: float = 0.9,
    coverage_weight: float = 5.0,
) -> LossBreakdown:
    """Differentiable objective balancing cover size and soft edge feasibility."""
    if not 0.0 <= coverage_target <= 1.0:
        raise ValueError("coverage_target must be in [0, 1]")
    if coverage_weight < 0:
        raise ValueError("coverage_weight must be non-negative")

    scores = node_scores.view(-1)
    size_loss = scores.mean() if scores.numel() else scores.new_zeros(())

    edges = unique_undirected_edges(edge_index)
    if edges.numel() == 0:
        coverage_loss = scores.new_zeros(())
    else:
        u, v = edges
        coverage_probability = scores[u] + scores[v] - scores[u] * scores[v]
        coverage_loss = F.relu(coverage_target - coverage_probability).mean()

    total_loss = size_loss + coverage_weight * coverage_loss
    return LossBreakdown(total=total_loss, size=size_loss, coverage=coverage_loss)


def train_gnn_for_vertex_cover(
    model: nn.Module,
    graph_data: Data,
    num_epochs: int = 200,
    lr: float = 1e-3,
    coverage_target: float = 0.9,
    coverage_weight: float = 5.0,
    verbose: bool = True,
) -> nn.Module:
    """Train the model on one graph using the soft vertex-cover objective."""
    if num_epochs < 0:
        raise ValueError("num_epochs must be non-negative")
    if lr <= 0:
        raise ValueError("lr must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        node_scores = model(graph_data.x, graph_data.edge_index).view(-1)
        losses = vertex_cover_loss(
            node_scores,
            graph_data.edge_index,
            coverage_target=coverage_target,
            coverage_weight=coverage_weight,
        )
        losses.total.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Loss={losses.total.item():.4f}, "
                f"Size={losses.size.item():.4f}, "
                f"Coverage={losses.coverage.item():.4f}"
            )

    return model


def is_vertex_cover(edge_index: torch.Tensor, selected: Iterable[int]) -> bool:
    """Return True when every undirected edge has at least one selected endpoint."""
    selected_set = set(int(node) for node in selected)
    edges = unique_undirected_edges(edge_index)
    for u, v in edges.t().tolist():
        if u not in selected_set and v not in selected_set:
            return False
    return True


def repair_and_prune_vertex_cover(
    node_scores: torch.Tensor,
    edge_index: torch.Tensor,
    threshold: float = 0.5,
) -> set[int]:
    """Threshold scores, repair uncovered edges, then prune redundant vertices.

    The returned set is guaranteed to be a valid vertex cover for ``edge_index``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    scores = node_scores.detach().view(-1).cpu()
    edges = unique_undirected_edges(edge_index.detach().cpu())
    selected = {i for i, score in enumerate(scores.tolist()) if score > threshold}

    # Repair: every uncovered edge receives its higher-scoring endpoint.
    for u, v in edges.t().tolist():
        if u not in selected and v not in selected:
            if scores[u].item() >= scores[v].item():
                selected.add(u)
            else:
                selected.add(v)

    # Prune low-confidence vertices first when their removal keeps feasibility.
    for node in sorted(selected, key=lambda idx: (scores[idx].item(), idx)):
        candidate = selected - {node}
        if is_vertex_cover(edges, candidate):
            selected = candidate

    return selected


def extract_vertex_cover(
    model: nn.Module, graph_data: Data, threshold: float = 0.5
) -> set[int]:
    """Run inference and return a guaranteed-feasible repaired vertex cover."""
    model.eval()
    with torch.no_grad():
        node_scores = model(graph_data.x, graph_data.edge_index).view(-1)

    vertex_cover = repair_and_prune_vertex_cover(
        node_scores=node_scores,
        edge_index=graph_data.edge_index,
        threshold=threshold,
    )

    valid = is_vertex_cover(graph_data.edge_index, vertex_cover)
    print(f"Is the solution a valid vertex cover? {valid}")
    print(f"Vertex cover size: {len(vertex_cover)} out of {node_scores.numel()} nodes")
    return vertex_cover


def visualize_solution(graph: nx.Graph, vertex_cover: set[int]) -> None:
    """Visualize selected vertices in red and non-selected vertices in light blue."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    non_cover_nodes = [n for n in graph.nodes() if n not in vertex_cover]
    nx.draw_networkx_nodes(
        graph, pos, nodelist=non_cover_nodes, node_color="lightblue", node_size=300
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=list(vertex_cover), node_color="red", node_size=300
    )
    nx.draw_networkx_labels(graph, pos)

    plt.title("Vertex Cover Solution (Red Nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    graph = generate_random_graph(20, edge_probability=0.2, seed=42)
    graph_data = networkx_to_pytorch_geometric(graph)
    print(
        f"Graph has {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges"
    )

    model = VertexCoverGNN(hidden_channels=64, num_layers=3)
    train_gnn_for_vertex_cover(model, graph_data, num_epochs=300)

    vertex_cover = extract_vertex_cover(model, graph_data, threshold=0.5)

    approximation_cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph)
    print(f"NetworkX approximation cover size: {len(approximation_cover)}")

    visualize_solution(graph, vertex_cover)


if __name__ == "__main__":
    main()
