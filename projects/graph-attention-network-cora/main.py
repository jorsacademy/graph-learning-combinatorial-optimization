import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import k_hop_subgraph, to_networkx


class GAT(nn.Module):
    """Two-layer Graph Attention Network for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


def accuracy(log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = log_probs.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_epoch(model, data, optimizer) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item(), train_acc


@torch.no_grad()
def evaluate(model, data, mask) -> tuple[float, float]:
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[mask], data.y[mask]).item()
    acc = accuracy(output[mask], data.y[mask])
    return loss, acc


def plot_training(losses: list[float], accuracies: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    axes[1].plot(accuracies)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")

    fig.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_attention(model: GAT, data, node_idx: int = 0) -> None:
    """Visualize first-layer attention around one node using a 1-hop subgraph."""
    model.eval()

    if node_idx < 0 or node_idx >= data.num_nodes:
        raise ValueError(f"node_idx must be between 0 and {data.num_nodes - 1}.")

    _, (att_edge_index, alpha) = model.gat1(
        data.x,
        data.edge_index,
        return_attention_weights=True,
    )

    # Average attention over heads to obtain one weight per directed edge.
    alpha = alpha.mean(dim=1)

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx,
        num_hops=1,
        edge_index=att_edge_index,
        relabel_nodes=False,
    )
    subset_set = set(subset.tolist())

    graph = nx.DiGraph()
    for node in subset.tolist():
        graph.add_node(node)

    for edge_pos in range(att_edge_index.size(1)):
        src = int(att_edge_index[0, edge_pos])
        dst = int(att_edge_index[1, edge_pos])
        if src in subset_set and dst in subset_set and (src == node_idx or dst == node_idx):
            graph.add_edge(src, dst, weight=float(alpha[edge_pos].cpu()))

    if graph.number_of_edges() == 0:
        print(f"No attention edges found for node {node_idx}.")
        return

    pos = nx.spring_layout(graph, seed=42)
    node_colors = ["red" if node == node_idx else "lightblue" for node in graph.nodes()]
    edge_widths = [1.0 + 8.0 * graph[u][v]["weight"] for u, v in graph.edges()]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        alpha=0.7,
        arrows=True,
        arrowsize=14,
    )
    nx.draw_networkx_labels(graph, pos)
    plt.title(f"First-layer attention around Cora node {node_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_dataset(data, max_nodes: int = 500) -> None:
    """Visualize a bounded sample of the graph to keep plotting responsive."""
    graph = to_networkx(data, to_undirected=True)

    if graph.number_of_nodes() > max_nodes:
        sampled_nodes = sorted(random.sample(list(graph.nodes()), max_nodes))
        graph = graph.subgraph(sampled_nodes).copy()

    labels = [int(data.y[node].cpu()) for node in graph.nodes()]
    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=35,
        node_color=labels,
        cmap="tab10",
    )
    nx.draw_networkx_edges(graph, pos, width=0.3, alpha=0.4)
    plt.title(f"Cora graph sample ({graph.number_of_nodes()} nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GAT on the Cora citation network.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--node", type=int, default=0, help="Node used for attention visualization.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    losses: list[float] = []
    train_accuracies: list[float] = []

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, data, optimizer)
        losses.append(loss)
        train_accuracies.append(train_acc)

        if epoch == 1 or epoch % 10 == 0:
            val_loss, val_acc = evaluate(model, data, data.val_mask)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    val_loss, val_acc = evaluate(model, data, data.val_mask)
    test_loss, test_acc = evaluate(model, data, data.test_mask)

    print(f"Validation loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")
    print(f"Test loss:       {test_loss:.4f} | accuracy: {test_acc:.4f}")

    if not args.no_plots:
        # Move graph data back to CPU for NetworkX/Matplotlib visualization.
        cpu_data = data.cpu()
        model = model.cpu()
        plot_training(losses, train_accuracies)
        visualize_attention(model, cpu_data, node_idx=args.node)
        visualize_dataset(cpu_data)


if __name__ == "__main__":
    main()
