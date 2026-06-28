import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.utils.logger import setup_logger

logger = setup_logger("GraphBuilder")


class TransactionGraphBuilder:
    """
    Converts a tabular transaction DataFrame into an inductive PyTorch
    Geometric graph for GraphSAGE discriminator consumption.

    Node semantics:  Each unique card1 entity becomes a graph node.
    Edge semantics:  Two types of directed edges are constructed:
      1. **Temporal self-edges**: model intra-card state evolution over time.
      2. **Cross-card risk edges**: link card entities that share the same
         P_emaildomain or addr1, propagating risk signals across structurally
         related accounts (catches carousel / ring fraud patterns).

    Node features are aggregated per card via mean pooling — this is the
    standard inductive GNN approach for heterogeneous transaction counts.
    All feature aggregation uses pd.concat to prevent DataFrame fragmentation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.node_feature_cols = config["features"]["gnn_node_features"]

    def build_inductive_graph(self, df: pd.DataFrame) -> Data:
        """
        Transforms a processed IEEE-CIS DataFrame into a PyTorch Geometric
        Data object with node features, labels, and multi-type edges.

        Args:
            df: Preprocessed + feature-engineered transaction DataFrame.
                Must contain: card1, isFraud, TransactionDT, V-features,
                              card1_count_cumulative, amount_to_mean_ratio.

        Returns:
            PyTorch Geometric Data object ready for GNN training.
        """
        logger.info(
            "Initializing inductive graph construction from tabular transactions..."
        )

        # ----------------------------------------------------------------
        # 1. Build Node Feature Matrix
        # ----------------------------------------------------------------
        # Dynamically resolve available V-features (may vary by subset size)
        v_features = sorted(
            [
                c
                for c in df.columns
                if c.startswith("V") and not c.endswith("_is_missing")
            ]
        )
        engineered_cols = ["card1_count_cumulative", "amount_to_mean_ratio"]
        all_feature_cols = v_features + self.node_feature_cols + engineered_cols

        # Keep only columns that actually exist in the current DataFrame
        all_feature_cols = [c for c in all_feature_cols if c in df.columns]

        logger.info(
            f"Aggregating {len(all_feature_cols)} features per card node via mean pooling..."
        )

        # Map card entities to sequential integer node indices
        unique_cards = df["card1"].unique()
        card_to_idx = {card: idx for idx, card in enumerate(unique_cards)}

        # Aggregate node features per card (mean over all transactions per card)
        # Using reindex(unique_cards) preserves the same ordering as card_to_idx
        node_features_df = (
            df.groupby("card1")[all_feature_cols].mean().reindex(unique_cards)
        )

        # Node label: a card node is fraudulent if ANY of its transactions was fraud
        node_labels_df = df.groupby("card1")["isFraud"].max().reindex(unique_cards)

        # Convert to PyTorch tensors
        x = torch.tensor(node_features_df.values, dtype=torch.float32)
        y = torch.tensor(node_labels_df.values, dtype=torch.long)

        # ----------------------------------------------------------------
        # 2. Construct Temporal Self-Edges (intra-card chronological links)
        # ----------------------------------------------------------------
        logger.info("Building temporal transaction sequence edges per card...")
        temporal_edges = []

        for card, group in df.groupby("card1"):
            if len(group) > 1:
                card_node_idx = card_to_idx[card]
                # Self-loop edges model the evolving state of a card entity over time
                n_steps = len(group) - 1
                for _ in range(n_steps):
                    temporal_edges.append([card_node_idx, card_node_idx])

        # ----------------------------------------------------------------
        # 3. Construct Cross-Card Risk Propagation Edges
        # ----------------------------------------------------------------
        # Two cards sharing the same email domain or address likely belong to
        # the same fraud ring — linking them propagates risk signals across
        # structurally related accounts for richer GNN message passing.
        logger.info(
            "Building cross-card risk propagation edges (email/address sharing)..."
        )
        cross_edges = []

        for link_col in ["P_emaildomain", "addr1"]:
            if link_col not in df.columns:
                continue

            # Get the modal (most frequent) value per card for this attribute
            card_attr = (
                df.groupby("card1")[link_col]
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
                .dropna()
            )

            # Group cards by shared attribute value
            for attr_val, cards_with_attr in card_attr.groupby(card_attr):
                card_list = cards_with_attr.index.tolist()
                if len(card_list) < 2:
                    continue  # No shared-attribute edge possible with only one card

                # Limit fan-out to top 10 co-members to prevent hub-node over-connection
                card_list = card_list[:10]

                # Create bidirectional edges between all co-members (pairwise)
                for i in range(len(card_list)):
                    for j in range(i + 1, len(card_list)):
                        src = card_to_idx.get(card_list[i])
                        dst = card_to_idx.get(card_list[j])
                        if src is not None and dst is not None:
                            cross_edges.append([src, dst])
                            cross_edges.append([dst, src])  # Bidirectional

        # ----------------------------------------------------------------
        # 4. Combine and Validate Edge Index
        # ----------------------------------------------------------------
        all_edges = temporal_edges + cross_edges

        if len(all_edges) == 0:
            logger.warning(
                "No edges constructed — falling back to self-loop identity edges. "
                "This typically means the DataFrame subset has only single-transaction cards."
            )
            edge_index = (
                torch.arange(x.size(0), dtype=torch.long).unsqueeze(0).repeat(2, 1)
            )
        else:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

        # ----------------------------------------------------------------
        # 5. Wrap into PyTorch Geometric Data
        # ----------------------------------------------------------------
        graph_data = Data(x=x, edge_index=edge_index, y=y)

        # Log class distribution for fraud auditing
        n_fraud = int(y.sum().item())
        n_total = int(y.size(0))
        logger.info(
            f"Graph construction complete — "
            f"Nodes: {graph_data.num_nodes:,} | "
            f"Edges: {graph_data.num_edges:,} "
            f"(temporal: {len(temporal_edges):,}, cross-card: {len(cross_edges):,}) | "
            f"Fraud nodes: {n_fraud:,} / {n_total:,} ({100 * n_fraud / n_total:.2f}%)"
        )
        return graph_data
