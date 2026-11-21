import numpy as np
import pytest
from src.graph.subgraph_engine import RealTimeGraphEngine


def test_graph_engine_initialization():
    engine = RealTimeGraphEngine()
    stats = engine.get_stats()
    assert stats["total_nodes"] >= 15
    assert stats["tracked_cards"] >= 15


def test_cold_start_subgraph_extraction():
    engine = RealTimeGraphEngine()
    dummy_feats = np.zeros(345, dtype=np.float32)
    dummy_feats[339] = 120.0

    # Cold start card with no history
    x_sub, edge_index, target_idx = engine.extract_subgraph(
        card1=999999,
        target_features=dummy_feats,
        p_emaildomain="unknown.org",
    )

    assert target_idx == 0
    assert x_sub.shape == (1, 345)
    assert edge_index.shape == (2, 1)


def test_syndicate_ring_subgraph_extraction():
    engine = RealTimeGraphEngine()
    dummy_feats = np.zeros(345, dtype=np.float32)
    dummy_feats[339] = 250.0

    # Known syndicate card 19901 sharing protonmail.com
    x_sub, edge_index, target_idx = engine.extract_subgraph(
        card1=19901,
        target_features=dummy_feats,
        p_emaildomain="protonmail.com",
    )

    assert target_idx == 0
    assert x_sub.shape[0] > 1  # Extracted multi-node neighborhood!
    assert edge_index.shape[1] > 0  # Extracted real topological edges!


def test_add_transaction_updates_graph():
    engine = RealTimeGraphEngine(max_history_nodes=50)
    dummy_feats = np.ones(345, dtype=np.float32)

    initial_nodes = engine.get_stats()["total_nodes"]
    engine.add_transaction(
        tx_id="NEW_TX_001",
        card1=77777,
        target_features=dummy_feats,
        p_emaildomain="fintech.com",
    )

    new_stats = engine.get_stats()
    assert new_stats["total_nodes"] == initial_nodes + 1
    assert 77777 in engine.card_to_nodes

