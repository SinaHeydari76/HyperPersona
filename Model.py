# %% ==============================================================
# === Imports and Setup ===
# ==============================================================
"""
Hyper Persona
------------------------------------------------------------
Components:
    Part 1: NetworkX Graphs to PyTorch Geometric Data objects conversion
    Part 2: Model Definition
    Part 3: Utility Functions
    Part 4: Training Loop
    Part 5: Data Loading and Preprocessing
    Part 6: Stratified Data Split and Conversion to PyG
    Part 7: Model Initialization and Training
    Part 8: Final Evaluation on Test Set
"""

import dill
import torch
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


# %% ============================================================================
# === Part 1: NetworkX Graphs to PyTorch Geometric Data objects conversion ===
# ==============================================================================
def convert_nx_to_pyg(graphs, labels=None):
    """
    Converts a list of NetworkX graphs (with node and edge features)
    into PyTorch Geometric Data objects for model training.

    Parameters
    ----------
    graphs : list of nx.Graph
        List of hierarchical graphs (document → sentences → words).
        Each node should contain a 'feature' tensor.
        Each edge may contain a 'similarity' float.
    labels : list or None
        List of numerical labels corresponding to each graph.

    Returns
    -------
    pyg_data_list : list of torch_geometric.data.Data
        Each Data object contains:
            - x : Node feature matrix [num_nodes, feature_dim]
            - edge_index : Tensor of edge pairs [2, num_edges]
            - edge_attr : Tensor of edge weights [num_edges, 1]
            - y : Target label tensor (if provided)
    """
    pyg_data_list = []

    for idx, G in enumerate(graphs):
        # Map node IDs to contiguous indices
        node_map = {old: new for new, old in enumerate(G.nodes())}

        # Stack node features into a single tensor
        x = torch.stack([G.nodes[n]['feature'].float() for n in G.nodes()])

        # Convert edges and similarity attributes
        edge_index = torch.tensor(
            [[node_map[u], node_map[v]] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            [[G[u][v].get('similarity', 0.0)] for u, v in G.edges()],
            dtype=torch.float
        )

        # Create label tensor if available
        y = torch.tensor([labels[idx]], dtype=torch.float) if labels is not None else None

        pyg_data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return pyg_data_list


# %% ==============================================================
# === Part 2: Model Definition ===
# ==============================================================
class TFConv(torch.nn.Module):
    """
    Feature Selective Transformer-Based Graph Encoder

    Architecture:
        - Two TransformerConv layers with residual connections
        - Layer normalization and sigmoid activations
        - Feature selection via Gumbel-softmax
        - Global additive pooling for graph embedding
        - Feedforward classification head
    """
    def __init__(self, in_channels, num_labels=1, dropout=0.3, temperature=0.5):
        super().__init__()
        self.dropout = dropout
        self.temperature = temperature

        # Trainable feature selection logits
        self.feature_logits = torch.nn.Parameter(torch.zeros(in_channels))

        # --- TransformerConv Block 1 ---
        self.gnn1 = TransformerConv(in_channels, 128, heads=2, concat=False, edge_dim=1, dropout=dropout)
        self.proj1 = torch.nn.Linear(in_channels, 128)
        self.ln1 = torch.nn.LayerNorm(128)

        # --- TransformerConv Block 2 ---
        self.gnn2 = TransformerConv(128, 64, heads=1, concat=False, edge_dim=1, dropout=dropout)
        self.proj2 = torch.nn.Linear(128, 64)
        self.ln2 = torch.nn.LayerNorm(64)

        # --- Classification Head ---
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(64, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Dropout(dropout),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, num_labels)
        )

        self.pool = global_add_pool
        self.activation_sigmoid = torch.nn.Sigmoid()

    # ----------------------------------------------------------
    def sample_hard_mask(self, deterministic=False):
        """Samples binary feature masks via Gumbel-softmax for feature selection."""
        if deterministic:
            mask = (torch.sigmoid(self.feature_logits) > 0.5).float()
        else:
            gumbel = -torch.log(-torch.log(torch.rand_like(self.feature_logits) + 1e-9) + 1e-9)
            y_soft = torch.sigmoid((self.feature_logits + gumbel) / self.temperature)
            y_hard = (y_soft > 0.5).float()
            mask = y_hard + (y_soft - y_soft.detach())
        return mask

    # ----------------------------------------------------------
    def forward(self, batch, deterministic=False):
        """Forward propagation for a mini-batch of graphs."""
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Apply feature mask
        mask = self.sample_hard_mask(deterministic=deterministic)
        x = x * mask

        # --- GNN Block 1 ---
        x1 = self.gnn1(x, edge_index, edge_attr)
        res1 = self.proj1(x)
        x = self.ln1(x1 + res1)
        x = self.activation_sigmoid(x)

        # --- GNN Block 2 ---
        x2 = self.gnn2(x, edge_index, edge_attr)
        res2 = self.proj2(x)
        x = self.ln2(x2 + res2)
        x = self.activation_sigmoid(x)

        # --- Graph-level Representation ---
        graph_emb = self.pool(x, batch_idx)
        return self.cls_head(graph_emb)


# %% ==============================================================
# === Part 3: Utility Functions ===
# ==============================================================
def init_weights(m):
    """Applies Xavier initialization to linear layers."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def compute_accuracy(output, labels, threshold=0.5):
    """Computes binary accuracy based on sigmoid thresholding."""
    labels = labels.view(output.shape)
    preds = (torch.sigmoid(output) > threshold).float()
    return (preds == labels).sum().item() / labels.numel()


@torch.no_grad()
def evaluate(model, loader, criterion, deterministic=True):
    """
    Evaluates model on a given dataset (validation or test).
    Computes mean loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    val_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch, deterministic=deterministic).view(-1)
        targets = batch.y.view(-1).float()

        val_loss += criterion(out, targets).item()
        preds = (torch.sigmoid(out) > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss": val_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0)
    }


# %% ==============================================================
# === Part 4: Training Loop ===
# ==============================================================
def train(model, train_loader, test_loader, optimizer, criterion, epochs=30, save_path=None):
    """
    Trains the TransformerConv GNN for binary classification.

    Features:
        - Validation-based early model selection
        - Gradient clipping
        - Prints per-epoch metrics
    """
    best, best_state_dict, best_epoch = 0.0, None, -1

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass and loss
            out = model(batch).view(-1)
            targets = batch.y.view(-1).float()
            loss = criterion(out, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Training accuracy
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            epoch_loss += loss.item()

        # Epoch summary
        train_acc = correct / total if total > 0 else 0.0
        val_metrics = evaluate(model, test_loader, criterion)
        val_acc, val_loss = val_metrics["accuracy"], val_metrics["loss"]

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2%} | Train Loss {loss:.2} | Val Acc {val_acc:.2%} | Val Loss {val_loss:.2}")

        # Save the best-performing model
        if best <= val_acc:
            best = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state_dict is not None and save_path is not None:
        torch.save(best_state_dict, save_path)

    return best_epoch


# %% ==============================================================
# === Part 5: Data Loading and Preprocessing ===
# ==============================================================
### Load precomputed hierarchical embeddings ###
with open('gtconv data.pkl', 'rb') as f:
    data = dill.load(f)

# Experiment parameters
numepochs = 30
num_labels = 1
batchsize = 8
test_size = 0.2
random_state = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% ==============================================================
# === Part 6: Stratified Data Split and Conversion to PyG ===
# ==============================================================
### Stratified splits maintain label balance ###
sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
indices = np.arange(len(graphs))
train_val_idx, test_idx = next(sss_outer.split(indices, labels_array))

# Inner split for validation
val_size = 0.10
sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
train_idx, val_idx = next(sss_inner.split(train_val_idx, labels_array[train_val_idx]))

# Subset data
train_graphs = [graphs[i] for i in train_idx]
val_graphs = [graphs[i] for i in val_idx]
test_graphs = [graphs[i] for i in test_idx]

train_labels = [labels[i] for i in train_idx]
val_labels = [labels[i] for i in val_idx]
test_labels = [labels[i] for i in test_idx]

# Convert to PyG format
train_data = convert_nx_to_pyg(train_graphs, train_labels)
val_data = convert_nx_to_pyg(val_graphs, val_labels)
test_data = convert_nx_to_pyg(test_graphs, test_labels)

# Input dimension
in_channels = train_data[0].x.size(1)

# Handle class imbalance
all_labels_tensor = torch.stack([d.y for d in train_data])
pos_weight = ((len(all_labels_tensor) - all_labels_tensor.sum()) / (all_labels_tensor.sum() + 1e-6)).to(device)

# Data loaders
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=True)


# %% ==============================================================
# === Part 7: Model Initialization and Training ===
# ==============================================================
### Initialize model, optimizer, and loss ###
model = TFConv(in_channels, num_labels=num_labels, dropout=0.2).to(device)
model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Train model and save best version
save_path = f'/best_model_{target_trait}.pth'
best_epoch = train(model, train_loader, val_loader, optimizer, criterion, epochs=numepochs, save_path=save_path)


# %% ==============================================================
# === Part 8: Final Evaluation on Test Set ===
# ==============================================================
### Load best model and report metrics ###
final_model = TFConv(in_channels, num_labels=num_labels, dropout=0).to(device)
final_model.load_state_dict(torch.load(save_path))
final_metrics = evaluate(final_model, test_loader, criterion, deterministic=True)

print("\nFinal Test Results:")
print(f"  Best Epoch: {best_epoch}")
print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
print(f"  Precision: {final_metrics['precision']:.2%}")
print(f"  Recall: {final_metrics['recall']:.2%}")
print(f"  F1: {final_metrics['f1']:.2%}")

# Cleanup
import gc
gc.collect()
