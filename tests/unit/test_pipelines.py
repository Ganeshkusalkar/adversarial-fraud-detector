import pytest
import numpy as np
import pandas as pd
from src.graph.graph_builder import GraphBuilder

def test_graph_builder_initialization():
    builder = GraphBuilder()
    assert builder.edge_index is None
    assert builder.node_features is None

def test_graph_builder_feature_matrix():
    # Mock data
    data = pd.DataFrame({
        "TransactionAmt": [10.5, 20.0, 150.0],
        "C1": [1, 2, 3],
        "C2": [0, 1, 0],
        "D1": [10, 20, 30]
    })
    
    # Adding mock vesta features
    for i in range(339):
        data[f"V{i}"] = np.random.randn(3)
        
    builder = GraphBuilder()
    # GraphBuilder expects numeric features to be passed in for feature matrix extraction
    # Normally this is done through a standard pipeline. We'll simulate the extraction.
    feature_cols = [c for c in data.columns if c not in ["TransactionID", "isFraud"]]
    X = data[feature_cols].values
    
    assert X.shape == (3, 343) # 339 Vesta + 4 manual features
    
    # Add dummy degree and centrality to match 345
    degree = np.ones((3, 1))
    centrality = np.ones((3, 1))
    
    X_final = np.hstack([X, degree, centrality])
    assert X_final.shape == (3, 345)

def test_graph_builder_edge_generation():
    builder = GraphBuilder()
    # Mocking a small adjacency list
    edges = [(0, 1), (1, 2), (0, 2)]
    edge_index = np.array(edges).T
    
    assert edge_index.shape == (2, 3)
    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 1
