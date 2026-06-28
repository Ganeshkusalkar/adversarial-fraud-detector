import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import dropout_edge
from src.utils.logger import setup_logger

logger = setup_logger("DiscriminatorGNN")


class FraudGNN(nn.Module):
    """
    Production GraphSAGE network for inductive fraud node classification.

    Architecture improvements over baseline:
      - **BatchNorm** after each SAGEConv layer for training stability on
        imbalanced data (prevents fraud-class signal from being washed out).
      - **Residual skip connection** from layer-1 output to layer-3 input,
        preserving local neighborhood features alongside deep structural context.
      - **Configurable decision threshold** exposed for recall tuning at
        inference time without retraining.

    3-hop message passing captures:
      - Hop 1: Direct transaction counterparties (immediate fraud signals)
      - Hop 2: Merchant / card network structural splits
      - Hop 3: Multi-layer money-loop and carousel fraud patterns
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        dropout: float = 0.3,
        decision_threshold: float = 0.38,
    ):
        """
        Args:
            in_channels:         Dimension of input node feature vectors.
            hidden_channels:     Hidden representation dimension.
            out_channels:        Output class count (2: legitimate / fraudulent).
            dropout:             Dropout rate for regularization.
            decision_threshold:  Fraud probability cutoff used during inference.
                                 Lowered to 0.38 (vs default 0.5) to improve
                                 recall by tolerating a controlled precision trade-off.
        """
        super().__init__()
        logger.info(
            f"Initializing FraudGNN: in={in_channels} → hidden={hidden_channels} "
            f"→ out={out_channels} | dropout={dropout} | threshold={decision_threshold}"
        )

        self.decision_threshold = decision_threshold
        half_hidden = hidden_channels // 2

        # --- Layer 1: 1-hop local neighborhood aggregation ---
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.LayerNorm(hidden_channels)

        # --- Layer 2: 2-hop structural pattern aggregation ---
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.LayerNorm(hidden_channels)

        # --- Layer 3: 3-hop macro context (reduced to half_hidden) ---
        self.conv3 = SAGEConv(hidden_channels, half_hidden)
        self.bn3 = nn.LayerNorm(half_hidden)

        # --- Skip connection projection ---
        # Projects layer-1 output (hidden_channels) to match layer-3 output (half_hidden)
        # so they can be summed. Learned projection preserves early local features.
        self.skip_proj = nn.Linear(hidden_channels, half_hidden, bias=False)

        # --- Output classifier ---
        self.classifier = nn.Linear(half_hidden, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Weight initialization: Kaiming He for ReLU networks
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.skip_proj.weight, nonlinearity="linear")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Runs inductive message passing and returns classification logits.

        Args:
            x:          Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch:      Batch vector for multi-graph pooling [num_nodes] (optional)

        Returns:
            Logits tensor [num_nodes, out_channels] (or [num_graphs, out_channels])
        """
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.3)

        # --- Layer 1: Local neighborhood features ---
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # --- Layer 2: 2-hop structural features ---
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        # --- Layer 3: 3-hop macro context features ---
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)

        # --- Residual skip connection: Layer-1 → Layer-3 ---
        # Adds local neighborhood signal back into the deep representation,
        # preventing over-smoothing and retaining discriminative local features
        skip = self.skip_proj(h1)
        h_combined = h3 + skip  # Element-wise residual addition

        # --- Multi-graph pooling (batch mode only) ---
        if batch is not None:
            h_combined = global_mean_pool(h_combined, batch)

        # --- Classification head ---
        return self.classifier(h_combined)

    @torch.no_grad()
    def predict_fraud_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience inference method returning fraud probability scores.

        Returns:
            Tensor of shape [num_nodes] with fraud probability per node.
        """
        self.eval()
        logits = self.forward(x, edge_index)
        return torch.softmax(logits, dim=-1)[:, 1]

    @torch.no_grad()
    def classify(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns binary fraud predictions using the tuned decision threshold.

        Returns:
            Boolean tensor [num_nodes]: True = predicted fraud.
        """
        proba = self.predict_fraud_proba(x, edge_index)
        return proba >= self.decision_threshold


# Bugfix: replaced batchnorm with layernorm to fix test/train mismatch

# Feature: added dropedge regularization to gnn forward pass
