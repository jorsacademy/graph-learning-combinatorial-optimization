import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.convs[-1](x, edge_index)

    @torch.no_grad()
    def embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return hidden node representations before the classification layer."""
        self.eval()
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred == target).float().mean().item())


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

    train_acc = accuracy(pred[data.train_mask], data.y[data.train_mask])
    val_acc = accuracy(pred[data.val_mask], data.y[data.val_mask])
    test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])
    return train_acc, val_acc, test_acc


def train_model(model, data, optimizer, epochs: int = 200):
    train_losses = []
    val_accuracies = []

    best_val_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, val_acc, _ = evaluate(model, data)
        train_losses.append(float(loss.item()))
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch: {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    _, final_val_acc, final_test_acc = evaluate(model, data)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {final_val_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")

    return train_losses, val_accuracies


def plot_training(train_losses, val_accuracies) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_embeddings(model, data, num_classes: int) -> None:
    hidden = model.embeddings(data.x, data.edge_index).cpu().numpy()

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    embedded_2d = tsne.fit_transform(hidden)
    labels = data.y.cpu().numpy()

    plt.figure(figsize=(10, 8))
    for class_id in range(num_classes):
        idx = labels == class_id
        plt.scatter(
            embedded_2d[idx, 0],
            embedded_2d[idx, 1],
            label=f"Class {class_id}",
            alpha=0.7,
            s=35,
        )

    plt.legend()
    plt.title("GraphSAGE Node Embeddings on Cora (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


def explain_graphsage_on_small_graph() -> None:
    """Visualize 1-hop and 2-hop neighborhoods on Zachary's Karate Club graph."""
    graph = nx.karate_club_graph()
    pos = nx.spring_layout(graph, seed=SEED)

    club_to_class = {"Mr. Hi": 0, "Officer": 1}
    communities = [club_to_class[graph.nodes[n]["club"]] for n in graph.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=communities,
        cmap=plt.cm.coolwarm,
        with_labels=True,
        node_size=500,
        font_weight="bold",
    )
    plt.title("Zachary's Karate Club Graph")
    plt.axis("off")
    plt.show()

    target_node = 0
    one_hop = set(graph.neighbors(target_node))
    two_hop = set()
    for node in one_hop:
        two_hop.update(graph.neighbors(node))
    two_hop -= one_hop | {target_node}

    node_colors = []
    for node in graph.nodes:
        if node == target_node:
            node_colors.append("red")
        elif node in one_hop:
            node_colors.append("gold")
        elif node in two_hop:
            node_colors.append("green")
        else:
            node_colors.append("gray")

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        font_weight="bold",
    )
    plt.title(f"GraphSAGE Neighborhood for Node {target_node}")
    plt.plot([], [], "o", color="red", label="Target Node")
    plt.plot([], [], "o", color="gold", label="1-Hop Neighbors")
    plt.plot([], [], "o", color="green", label="2-Hop Neighbors")
    plt.plot([], [], "o", color="gray", label="Other Nodes")
    plt.legend(loc="lower right")
    plt.axis("off")
    plt.show()

    print("\nGraphSAGE message passing:")
    print(f"1. Target node {target_node} starts with its own feature vector.")
    print(f"2. Layer 1 aggregates from {len(one_hop)} one-hop neighbors.")
    print("3. Layer 2 lets information from two-hop neighbors influence the target node.")
    print("4. The learned aggregation function can be applied to unseen nodes or graphs.")


def print_aggregator_notes() -> None:
    print("\nGraphSAGE aggregator variants:")
    print("- Mean: element-wise mean of neighboring representations.")
    print("- GCN-style: combines self and neighbor information in a normalized aggregate.")
    print("- Pooling: transforms neighbors with an MLP before pooling.")
    print("- LSTM: aggregates an ordered/permuted neighbor sequence with an LSTM.")
    print("\nPyTorch Geometric SAGEConv uses mean aggregation by default.")


def main() -> None:
    set_seed()

    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]

    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Training nodes: {int(data.train_mask.sum())}")
    print(f"Validation nodes: {int(data.val_mask.sum())}")
    print(f"Test nodes: {int(data.test_mask.sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = data.to(device)
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    train_losses, val_accuracies = train_model(
        model=model,
        data=data,
        optimizer=optimizer,
        epochs=200,
    )

    plot_training(train_losses, val_accuracies)
    visualize_embeddings(model, data, dataset.num_classes)
    explain_graphsage_on_small_graph()
    print_aggregator_notes()

    print("\nInductive vs. transductive learning:")
    print("Transductive models are tied to the graph observed during training.")
    print("GraphSAGE learns neighborhood aggregation functions, enabling inductive use.")


if __name__ == "__main__":
    main()
