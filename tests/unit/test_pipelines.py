"""
Expanded unit tests for TransactionGraphBuilder.
Covers: node count correctness, feature matrix dimensions, label alignment,
edge generation, isolated-node fallback, and cross-card risk edges.
"""

import pytest
import numpy as np
import pandas as pd

# Skip entire module at collection time if torch / torch_geometric are absent.
# This prevents a hard ImportError from graph_builder.py which has top-level
# `import torch` and `from torch_geometric.data import Data`.
torch = pytest.importorskip(
    "torch", reason="PyTorch not installed — skipping graph builder tests"
)
pytest.importorskip(
    "torch_geometric",
    reason="torch_geometric not installed — skipping graph builder tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_df(n_cards=3, seed=42):
    """
    Build a minimal valid DataFrame for the GraphBuilder.
    Each card has exactly one transaction.
    """
    np.random.seed(seed)
    cards = list(range(1, n_cards + 1))
    data = {
        "card1": cards,
        "TransactionAmt": np.random.uniform(10, 500, n_cards),
        "C1": np.random.uniform(0, 5, n_cards),
        "C2": np.random.uniform(0, 3, n_cards),
        "D1": np.random.uniform(0, 30, n_cards),
        "isFraud": [0] * (n_cards - 1) + [1],
        "TransactionDT": list(range(n_cards)),
        "card1_count_cumulative": [1] * n_cards,
        "amount_to_mean_ratio": [1.0] * n_cards,
    }
    df = pd.DataFrame(data)
    for i in range(339):
        df[f"V{i}"] = np.random.randn(n_cards)
    return df


def _config(extra_features=None):
    feats = extra_features or ["TransactionAmt", "C1", "C2", "D1"]
    return {"features": {"gnn_node_features": feats}}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestGraphBuilderInit:
    def test_init_with_empty_node_features(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        builder = TransactionGraphBuilder({"features": {"gnn_node_features": []}})
        assert builder is not None
        assert builder.node_feature_cols == []

    def test_init_with_custom_features(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        builder = TransactionGraphBuilder(_config(["TransactionAmt"]))
        assert "TransactionAmt" in builder.node_feature_cols


# ---------------------------------------------------------------------------
# Node Count & Label Alignment
# ---------------------------------------------------------------------------


class TestGraphBuilderNodes:
    def test_node_count_equals_unique_cards(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=5)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.num_nodes == 5, f"Expected 5 nodes, got {graph.num_nodes}"

    def test_node_labels_shape_matches_nodes(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=4)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.y.shape == (4,), f"Labels shape mismatch: {graph.y.shape}"

    def test_fraud_label_propagated_to_card_node(self):
        """A card with at least one fraud txn must have label=1."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)
        # card 3 is flagged as fraud
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.y.max().item() == 1, "At least one fraud node must have label=1"
        assert (
            graph.y.min().item() == 0
        ), "At least one legitimate node must have label=0"

    def test_all_legitimate_labels(self):
        """All-legitimate transactions → all labels = 0."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)
        df["isFraud"] = 0
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.y.sum().item() == 0


# ---------------------------------------------------------------------------
# Feature Matrix Dimensions
# ---------------------------------------------------------------------------


class TestGraphBuilderFeatureMatrix:
    def test_feature_matrix_shape(self):
        """339 V-features + 4 manual + 2 engineered = 345 columns."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.x.shape == (3, 345), f"Expected (3, 345), got {graph.x.shape}"

    def test_feature_matrix_float32(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.x.dtype == torch.float32

    def test_feature_matrix_no_nan(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=4)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert not torch.isnan(
            graph.x
        ).any(), "Feature matrix must not contain NaN values"


# ---------------------------------------------------------------------------
# Edge Generation
# ---------------------------------------------------------------------------


class TestGraphBuilderEdges:
    def test_single_transaction_per_card_no_temporal_edges(self):
        """Cards with 1 transaction each produce no temporal edges."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)  # each card has 1 txn
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        # No temporal edges + no cross edges (no shared domain) → fallback self-loops
        # edge_index should still be valid shape (2, E)
        assert graph.edge_index.shape[0] == 2

    def test_multi_transaction_card_generates_temporal_edges(self):
        """A card with 3 transactions generates 2 temporal edges (self-loops)."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = pd.DataFrame(
            {
                "card1": [1, 1, 1],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "C1": [1, 2, 3],
                "C2": [0, 1, 0],
                "D1": [5, 10, 15],
                "isFraud": [0, 0, 1],
                "TransactionDT": [1, 2, 3],
                "card1_count_cumulative": [1, 2, 3],
                "amount_to_mean_ratio": [0.5, 1.0, 1.5],
            }
        )
        for i in range(339):
            df[f"V{i}"] = np.random.randn(3)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        # 1 unique card → 1 node; 2 temporal self-loops
        assert graph.num_nodes == 1
        assert graph.edge_index.shape[1] == 2

    def test_cross_card_edges_from_shared_email(self):
        """Cards sharing P_emaildomain must be connected with bidirectional edges."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = pd.DataFrame(
            {
                "card1": [10, 20, 30],
                "TransactionAmt": [50.0, 100.0, 200.0],
                "C1": [1, 2, 3],
                "C2": [0, 1, 0],
                "D1": [5, 10, 15],
                "isFraud": [0, 0, 1],
                "TransactionDT": [1, 2, 3],
                "card1_count_cumulative": [1, 1, 1],
                "amount_to_mean_ratio": [1.0, 1.0, 1.0],
                "P_emaildomain": [
                    "fraud@ring.com",
                    "fraud@ring.com",
                    "other@domain.com",
                ],
            }
        )
        for i in range(339):
            df[f"V{i}"] = np.random.randn(3)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        # Cards 10 and 20 share email → 2 cross-card edges (bidirectional)
        assert graph.edge_index.shape[1] >= 2

    def test_edge_index_shape_valid(self):
        """edge_index must always be shape [2, E]."""
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=5)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.ndim == 2

    def test_edge_index_dtype_long(self):
        from src.graph.graph_builder import TransactionGraphBuilder

        df = _base_df(n_cards=3)
        builder = TransactionGraphBuilder(_config())
        graph = builder.build_inductive_graph(df)
        assert graph.edge_index.dtype == torch.long
