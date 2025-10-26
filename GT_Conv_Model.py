# %% === Imports ===
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

# %% === Convert NetworkX to PyG ===
def convert_nx_to_pyg(graphs, labels=None):
    pyg_data_list = []
    for idx, G in enumerate(graphs):
        node_map = {old: new for new, old in enumerate(G.nodes())}
        x = torch.stack([G.nodes[n]['feature'].float() for n in G.nodes()])
        edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[G[u][v].get('similarity', 0.0)] for u, v in G.edges()], dtype=torch.float)
        y = torch.tensor([labels[idx]], dtype=torch.float) if labels is not None else None
        pyg_data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return pyg_data_list

# %% === Transformer Convolution Model ===
class TFConv(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1, dropout=0.3, temperature=0.5):
        super().__init__()
        self.dropout = dropout
        self.temperature = temperature
        self.feature_logits = torch.nn.Parameter(torch.zeros(in_channels))
        self.gnn1 = TransformerConv(in_channels, 128, heads=2, concat=False, edge_dim=1, dropout=dropout)
        self.proj1 = torch.nn.Linear(in_channels, 128)
        self.ln1 = torch.nn.LayerNorm(128)
        self.gnn2 = TransformerConv(128, 64, heads=1, concat=False, edge_dim=1, dropout=dropout)
        self.proj2 = torch.nn.Linear(128, 64)
        self.ln2 = torch.nn.LayerNorm(64)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(64, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Dropout(dropout),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, num_labels)
        )
        self.pool = global_add_pool
        self.activation_sigmoid = torch.nn.Sigmoid()

    def sample_hard_mask(self, deterministic=False):
        if deterministic:
            mask = (torch.sigmoid(self.feature_logits) > 0.5).float()
        else:
            gumbel = -torch.log(-torch.log(torch.rand_like(self.feature_logits) + 1e-9) + 1e-9)
            y_soft = torch.sigmoid((self.feature_logits + gumbel) / self.temperature)
            y_hard = (y_soft > 0.5).float()
            mask = y_hard + (y_soft - y_soft.detach())
        return mask

    def forward(self, batch, deterministic=False):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        mask = self.sample_hard_mask(deterministic=deterministic)
        x = x * mask
        x1 = self.gnn1(x, edge_index, edge_attr)
        res1 = self.proj1(x)
        x = self.ln1(x1 + res1)
        x = self.activation_sigmoid(x)
        x2 = self.gnn2(x, edge_index, edge_attr)
        res2 = self.proj2(x)
        x = self.ln2(x2 + res2)
        x = self.activation_sigmoid(x)
        graph_emb = self.pool(x, batch_idx)
        return self.cls_head(graph_emb)
    
    @torch.no_grad()
    def get_embeddings(self, loader, device):
        self.eval()
        all_embeds, all_labels = [], []
        for batch in loader:
            batch = batch.to(device)
            x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

            # Reuse the encoder layers (same as forward up to pooling)
            mask = self.sample_hard_mask()
            x = x * mask

            x1 = self.gnn1(x, edge_index, edge_attr)
            res1 = self.proj1(x)
            x = self.ln1(x1 + res1)
            x = torch.sigmoid(x)

            x2 = self.gnn2(x, edge_index, edge_attr)
            res2 = self.proj2(x)
            x = self.ln2(x2 + res2)
            x = torch.sigmoid(x)

            graph_emb = self.pool(x, batch_idx)
            all_embeds.append(graph_emb.cpu())
            all_labels.append(batch.y.cpu())

        return torch.cat(all_embeds), torch.cat(all_labels)

# %% === Utilities ===
def create_sentence_graph(hierarchical, doc_index, window_size=1):
    """
    Creates a sentence-level co-occurrence graph for a document.
    Sentences that are close in order are connected,
    and each edge is weighted by cosine similarity of their embeddings.
    """
    G = nx.Graph()
    doc_emb, sentence_list = hierarchical
    doc_id = f"doc_{doc_index}"

    # Add the document node
    G.add_node(doc_id, feature=doc_emb, type='document')

    # Add sentence nodes
    for sent_idx, (sent_emb, _) in enumerate(sentence_list):
        sent_id = f"sent_{sent_idx}"
        G.add_node(sent_id, feature=sent_emb, type='sentence')
        # connect each sentence to the document node
        G.add_edge(doc_id, sent_id, similarity=F.cosine_similarity(doc_emb, sent_emb, dim=0).item())

    # Add co-occurrence edges between nearby sentences
    for i, (sent_emb_i, _) in enumerate(sentence_list):
        for j in range(i + 1, min(i + window_size + 1, len(sentence_list))):
            sent_emb_j, _ = sentence_list[j]
            sim = F.cosine_similarity(sent_emb_i, sent_emb_j, dim=0).item()
            G.add_edge(f"sent_{i}", f"sent_{j}", similarity=sim)

    return G

def create_word_graph(hierarchical, doc_index, window_size=2):
    """
    Creates a word-level co-occurrence graph for a document.
    Words that appear within the specified window are connected,
    and each edge is weighted by cosine similarity of their embeddings.
    """
    G = nx.Graph()
    doc_emb, sentence_list = hierarchical
    doc_id = f"doc_{doc_index}"

    # Add the document node
    G.add_node(doc_id, feature=doc_emb, type='document')

    # Collect all words in document order
    all_words = []
    for sent_idx, (_, word_emb_list) in enumerate(sentence_list):
        for word_idx, word_emb in enumerate(word_emb_list):
            word_id = f"word_{sent_idx}_{word_idx}"
            all_words.append((word_id, word_emb))

    # Add word nodes
    for word_id, word_emb in all_words:
        G.add_node(word_id, feature=word_emb, type='word')
        # connect each word to the document itself
        G.add_edge(doc_id, word_id, similarity=F.cosine_similarity(doc_emb, word_emb, dim=0).item())

    # Add co-occurrence edges between nearby words
    for i, (w_i, emb_i) in enumerate(all_words):
        for j in range(i + 1, min(i + window_size + 1, len(all_words))):
            w_j, emb_j = all_words[j]
            sim = F.cosine_similarity(emb_i, emb_j, dim=0).item()
            G.add_edge(w_i, w_j, similarity=sim)

    return G

def create_graph(hierarchical, doc_index):
    G = nx.Graph()
    doc_emb, sentence_list = hierarchical
    doc_id = 'doc_' + doc_index
    G.add_node(doc_id, feature=doc_emb, type='document')
    for sent_idx, (sent_emb, word_emb_list) in enumerate(sentence_list):
        sent_id = f"sent_{sent_idx}"
        G.add_node(sent_id, feature=sent_emb, type='sentence')
        G.add_edge(doc_id, sent_id, similarity=F.cosine_similarity(doc_emb, sent_emb, dim=0).item())
        for word_idx, word_emb in enumerate(word_emb_list):
            word_id = f"word_{sent_idx}_{word_idx}"
            G.add_node(word_id, feature=word_emb, type='word')
            G.add_edge(sent_id, word_id, similarity=F.cosine_similarity(doc_emb, word_emb, dim=0).item())
    return G

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def compute_accuracy(output, labels, threshold=0.5):
    labels = labels.view(output.shape)
    preds = (torch.sigmoid(output) > threshold).float()
    return (preds == labels).sum().item() / labels.numel()

@torch.no_grad()
def evaluate_binary(model, loader, criterion, deterministic=True):
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

# %% === Training ===

def train_binary(model, train_loader, test_loader, optimizer, criterion, epochs=30, save_path=None):
    best, best_state_dict, best_epoch = 0.0, None, -1
    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            targets = batch.y.view(-1).float()
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            epoch_loss += loss.item()
        train_acc = correct / total if total > 0 else 0.0
        val_metrics = evaluate_binary(model, test_loader, criterion)
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2%} | Train Loss {loss:.2} | Val Acc {val_acc:.2%} | Val Loss {val_loss:.2}")
        if best <= val_acc:
            best = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
    if best_state_dict is not None and save_path is not None:
        torch.save(best_state_dict, save_path)
    return best_epoch

# %% === Loading data ===
with open('gtconv data.pkl', 'rb') as f:
    data = dill.load(f)

# %%
numepochs = 30

num_labels = 1
batchsize = 8

test_size = 0.2
    
random_state = 42

# %% === Create graphs ===
hierarchical_embeddings_list = data['hirarchical_embeddings']
labels_dict = { 'o': data['o'], 'c': data['c'], 'e': data['e'], 'a': data['a'], 'n': data['n'] }

target_trait = 'n'
labels = labels_dict[target_trait]
labels = [l[0] if isinstance(l, (list, tuple, np.ndarray)) else l for l in labels]
labels_array = np.array(labels)
# graphs = [create_graph(hierarchical_embeddings_list[i], str(i)) for i in tqdm(range(len(hierarchical_embeddings_list)))]
graphs = [create_word_graph(hierarchical_embeddings_list[i], str(i)) for i in tqdm(range(len(hierarchical_embeddings_list)))]
# graphs = [create_sentence_graph(hierarchical_embeddings_list[i], str(i)) for i in tqdm(range(len(hierarchical_embeddings_list)))]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% === Model Training ===
# Create a single stratified split into train / temp, then split temp into val and test
# Single stratified split for train and test
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
indices = np.arange(len(graphs))
train_idx, test_idx = next(sss.split(indices, labels_array))

# Build graph and label subsets
train_graphs = [graphs[i] for i in train_idx]
test_graphs = [graphs[i] for i in test_idx]

train_labels = [labels[i] for i in train_idx]
test_labels = [labels[i] for i in test_idx]

# Convert to PyG
train_data = convert_nx_to_pyg(train_graphs, train_labels)
test_data = convert_nx_to_pyg(test_graphs, test_labels)

in_channels = train_data[0].x.size(1)

# Handle class imbalance on training set
all_labels_tensor = torch.stack([d.y for d in train_data])
pos_weight = ((len(all_labels_tensor) - all_labels_tensor.sum()) / (all_labels_tensor.sum() + 1e-6)).to(device)

# Data loaders
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=True)

# Model setup
model = TFConv(in_channels, num_labels=num_labels, dropout=0.2).to(device)
model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# %%
save_path = f'1.model selection/best_model_{target_trait}.pth'
best_epoch = train_binary(model, train_loader, test_loader, optimizer, criterion, epochs=numepochs, save_path=save_path)

# %%
# Load the saved best model (deterministic evaluation)
final_model = TFConv(in_channels, num_labels=num_labels, dropout=0).to(device)
final_model.load_state_dict(torch.load(save_path))
final_metrics = evaluate_binary(final_model, test_loader, criterion, deterministic=True)

print("\nFinal Test Results:")
print(f"  Best Epoch: {best_epoch}")
print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
print(f"  Precision: {final_metrics['precision']:.2%}")
print(f"  Recall: {final_metrics['recall']:.2%}")
print(f"  F1: {final_metrics['f1']:.2%}")
# %%
import gc

gc.collect()