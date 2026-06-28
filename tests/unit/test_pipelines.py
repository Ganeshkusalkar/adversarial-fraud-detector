import pytest
import numpy as np
import pandas as pd
from src.graph.graph_builder import TransactionGraphBuilder

def test_graph_builder_initialization():
    builder = TransactionGraphBuilder({"features": {"gnn_node_features": []}})
    assert builder is not None

def test_graph_builder_feature_matrix():
    # Mock data
    data = pd.DataFrame(
        {
            "card1": [1, 2, 3],
            "TransactionAmt": [10.5, 20.0, 150.0],
            "C1": [1, 2, 3],
            "C2": [0, 1, 0],
            "D1": [10, 20, 30],
            "isFraud": [0, 0, 1],
            "TransactionDT": [1, 2, 3],
            "card1_count_cumulative": [1, 1, 1],
            "amount_to_mean_ratio": [1.0, 1.0, 1.0]
        }
    )

    # Adding mock vesta features
    for i in range(339):
        data[f"V{i}"] = np.random.randn(3)

    builder = TransactionGraphBuilder({"features": {"gnn_node_features": ["TransactionAmt", "C1", "C2", "D1"]}})
    
    graph_data = builder.build_inductive_graph(data)
    
    assert graph_data.x.shape == (3, 345)  # 339 Vesta + 4 manual features + 2 engineered
    assert graph_data.y.shape == (3,)

def test_graph_builder_edge_generation():
    # Mock temporal edges for a card with multiple transactions
    data = pd.DataFrame(
        {
            "card1": [1, 1, 2],
            "TransactionAmt": [10.5, 20.0, 150.0],
            "C1": [1, 2, 3],
            "C2": [0, 1, 0],
            "D1": [10, 20, 30],
            "isFraud": [0, 0, 1],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com"]
        }
    )
    for i in range(339):
        data[f"V{i}"] = np.random.randn(3)
        
    builder = TransactionGraphBuilder({"features": {"gnn_node_features": []}})
    graph_data = builder.build_inductive_graph(data)
    
    # 2 unique cards -> 2 nodes. Card 1 has 2 tx -> 1 temporal edge (self-loop).
    # Since P_emaildomain is present but only card 1 has it for multiple tx, no cross edges between DIFFERENT cards.
    assert graph_data.edge_index.shape[1] == 1
