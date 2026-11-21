# ============================================================
# DAY 1 — Adversarial Transaction Disguise Detector
# Data loading, transaction graph construction, baseline GNN
# Target: ~9 hours. Runs on CPU. No Neo4j needed.
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# STEP 1: LOAD DATA  (~1-2 hrs including exploration)
# ------------------------------------------------------------
# CHANGE THIS PATH to wherever you saved the Kaggle IEEE-CIS files.
# The competition gives you train_transaction.csv and train_identity.csv
# For Day 1 we only need train_transaction.csv — identity merge is optional polish.

DATA_PATH = "data/ieee_cis/ieee_preprocessed.csv"

print("Loading transaction data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean():.4f}")
print(df['isFraud'].value_counts())

# Quick sanity check on key columns used to build the graph
required_cols = ['TransactionID', 'isFraud', 'TransactionAmt', 'TransactionDT', 'card1']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Check your CSV matches the Kaggle IEEE-CIS schema.")

# ------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING  (~1 hr)
# ------------------------------------------------------------
# Keep it simple for Day 1. We'll use numeric features that exist
# in the real dataset. Adjust the list below to match what's in
# your actual CSV (IEEE-CIS has hundreds of columns, V1-V339 etc.)

# Pick a small, reliable feature set first — expand later if time allows
numeric_candidates = ['TransactionAmt', 'TransactionDT', 'card1', 'card2', 'card3',
                       'card5', 'addr1', 'addr2', 'dist1', 'dist2']
feature_cols = [c for c in numeric_candidates if c in df.columns]
print(f"\nUsing {len(feature_cols)} numeric features: {feature_cols}")

# Fill missing values — IEEE-CIS has a LOT of NaNs, this is expected
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: card4 (card type), card6 (debit/credit), P_emaildomain
cat_candidates = ['card4', 'card6', 'P_emaildomain', 'ProductCD']
cat_cols = [c for c in cat_candidates if c in df.columns]
for col in cat_cols:
    df[col] = df[col].fillna('missing')
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

cat_enc_cols = [c + '_enc' for c in cat_cols]
all_feature_cols = feature_cols + cat_enc_cols
print(f"Total features for model: {len(all_feature_cols)}")

# Scale numeric features (important — GNNs are sensitive to feature scale)
scaler = StandardScaler()
df[all_feature_cols] = scaler.fit_transform(df[all_feature_cols])

# ------------------------------------------------------------
# STEP 3: BUILD TRANSACTION GRAPH  (~1 hr)
# ------------------------------------------------------------
# Graph logic: each TRANSACTION is a node.
# Two transactions are connected by an edge if they share an
# identifying entity — same card1 (proxy for account) within
# a reasonable time window. This is how disguised/structured
# fraud shows up: many small linked transactions.
#
# NOTE: We skip Neo4j entirely. We build edges directly as a
# PyTorch tensor — this is faster to build and totally fine
# for a project of this size (avoids a full extra dependency).

fraud_count = len(df[df['isFraud'] == 1])
nonfraud_count = len(df[df['isFraud'] == 0])

print("Fraud:", fraud_count)
print("NonFraud:", nonfraud_count)
print("\nBuilding transaction graph edges...")


# To keep this CPU-feasible in 5 days, subsample if the dataset is huge.
MAX_NODES = 50000

if len(df) > MAX_NODES:

    fraud_df = df[df['isFraud'] == 1]
    nonfraud_df_full = df[df['isFraud'] == 0]

    remaining = MAX_NODES - len(fraud_df)

    if remaining > 0:
        nonfraud_df = nonfraud_df_full.sample(
            n=min(remaining, len(nonfraud_df_full)),
            random_state=42
        )

        df = pd.concat([fraud_df, nonfraud_df])

    else:
        print(
            f"Fraud samples ({len(fraud_df)}) exceed MAX_NODES ({MAX_NODES}). "
            "Sampling fraud cases instead."
        )

        df = fraud_df.sample(
            n=MAX_NODES,
            random_state=42
        )

    df = df.sort_values('TransactionDT').reset_index(drop=True)

print(f"Subsampled to {len(df)} transactions")
df = df.reset_index(drop=True)
df['node_idx'] = np.arange(len(df))
edges = []
# Group by card1 (shared account/card proxy) — connect transactions
# from the same card that happen close together in time.
for card_id, group in df.groupby('card1'):
    if len(group) < 2:
        continue
    group_sorted = group.sort_values('TransactionDT')
    idxs = group_sorted['node_idx'].values
    # Connect consecutive transactions from the same card (chain structure)
    for i in range(len(idxs) - 1):
        edges.append((idxs[i], idxs[i + 1]))
        edges.append((idxs[i + 1], idxs[i]))  # undirected

if len(edges) == 0:
    raise ValueError("No edges created — check that 'card1' has repeated values across rows.")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"Graph built: {len(df)} nodes, {edge_index.shape[1]} directed edges")

# Node features and labels
x = torch.tensor(df[all_feature_cols].values, dtype=torch.float)
y = torch.tensor(df['isFraud'].values, dtype=torch.long)

print(f"Node feature matrix: {x.shape}")
print(f"Labels: {y.shape}, fraud count: {y.sum().item()}")

# ------------------------------------------------------------
# STEP 4: TRAIN/VAL/TEST SPLIT  (masks, not separate graphs)
# ------------------------------------------------------------
num_nodes = x.shape[0]
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y.numpy()
)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.2, random_state=42, stratify=y.numpy()[train_idx]
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

print(f"\nTrain: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

# ------------------------------------------------------------
# STEP 5: BASELINE GNN MODEL — GraphSAGE  (~3-4 hrs)
# ------------------------------------------------------------
# We implement GraphSAGE message passing FROM SCRATCH using plain
# PyTorch (no torch_geometric dependency needed) so it's lighter
# to install and easier to modify later for the adversarial part.

class SimpleGraphSAGELayer(nn.Module):
    """One GraphSAGE-style layer: aggregate neighbor features (mean),
    concatenate with self features, then linear + ReLU."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index  # row = source, col = target
        num_nodes = x.shape[0]

        # Mean aggregation of neighbor features per node
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, col, x[row])
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, col, torch.ones(row.shape[0], device=x.device))
        degree = degree.clamp(min=1).unsqueeze(1)
        neighbor_mean = neighbor_sum / degree

        combined = torch.cat([x, neighbor_mean], dim=1)
        return F.relu(self.linear(combined))


class FraudGNN(nn.Module):
    """2-layer GraphSAGE fraud classifier. This is your DISCRIMINATOR
    for Day 2-3's adversarial training."""
    def __init__(self, in_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layer1 = SimpleGraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = SimpleGraphSAGELayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        h = self.layer1(x, edge_index)
        h = self.dropout(h)
        h = self.layer2(h, edge_index)
        h = self.dropout(h)
        out = self.classifier(h)
        return out


# ------------------------------------------------------------
# STEP 6: TRAINING LOOP
# ------------------------------------------------------------
device = torch.device('cpu')  # no GPU — this is fine, dataset is small enough
x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

model = FraudGNN(in_dim=x.shape[1], hidden_dim=64, num_classes=2).to(device)

# Class imbalance is severe (fraud is rare) — weight the loss
class_counts = torch.bincount(y[train_mask])
class_weights = (1.0 / class_counts.float())
class_weights = class_weights / class_weights.sum() * 2
print(f"\nClass weights (to handle imbalance): {class_weights}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("\nTraining baseline GNN (vanilla, NOT adversarially hardened)...")
best_val_auc = 0
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 99:
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
            val_auc = roc_auc_score(y[val_mask].numpy(), val_probs.numpy())
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'baseline_gnn_best.pt')
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

# ------------------------------------------------------------
# STEP 7: FINAL EVALUATION ON TEST SET
# ------------------------------------------------------------
model.load_state_dict(torch.load('baseline_gnn_best.pt'))
model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    test_probs = F.softmax(out[test_mask], dim=1)[:, 1]
    test_preds = out[test_mask].argmax(dim=1)
    test_auc = roc_auc_score(y[test_mask].numpy(), test_probs.numpy())

print(f"\n{'='*50}")
print(f"DAY 1 BASELINE RESULTS")
print(f"{'='*50}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y[test_mask].numpy(), test_preds.numpy(),
                             target_names=['Legitimate', 'Fraud']))

print("""
NEXT STEPS (Day 2):
- This baseline GNN is your 'vanilla' discriminator.
- Tomorrow you build a GAN Generator that creates adversarial
  fraud transactions designed to fool THIS model.
- Save this script's model (baseline_gnn_best.pt) — you'll load
  it as the starting point for adversarial training.
- Expected Day 1 result: baseline AUC around 0.75-0.85 is normal
  and totally fine. The interesting result comes Day 3 when you
  show how adversarial examples crush this baseline's performance,
  then how hardening recovers it.
""")


