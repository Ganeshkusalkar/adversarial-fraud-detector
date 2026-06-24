import pytest
import pandas as pd
from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.graph.graph_builder import TransactionGraphBuilder

def test_ieee_to_graph_builder_flow():
    """
    Validates end-to-end integration flow from Raw DataFrame -> Preprocessed Features -> GraphBuilder.
    """
    config = {
        "data": {
            "ieee": {
                "transaction_path": "data/raw/train_transaction_mock.csv",
                "identity_path": "data/raw/train_identity_mock.csv"
            }
        },
        "features": {
            "ieee_categorical": ["ProductCD"],
            "gnn_node_features": ["TransactionAmt", "C1", "C2", "D1"]
        }
    }
    
    pipeline = IEEECISPipeline(config)
    
    # Mock dataframes matching the new pipeline requirements
    df_txn = pd.DataFrame({
        "TransactionID": [1000, 1001],
        "card1": [101, 102],
        "TransactionAmt": [50.0, 100.0],
        "TransactionDT": [3600, 7200],
        "ProductCD": ["W", "C"],
        "card4": ["visa", "mastercard"],
        "card6": ["credit", "debit"],
        "C1": [1.0, 2.0],
        "C2": [1.0, 1.0],
        "D1": [0.0, 0.0],
        "isFraud": [0, 1],
        "vesta_features": [[0.0]*339, [0.0]*339]
    })
    
    # Expand vesta_features to individual columns start with 'V' to match preprocessing expectation
    for i in range(1, 340):
        df_txn[f"V{i}"] = [0.0, 0.0]
        
    df_id = pd.DataFrame({
        "TransactionID": [1000, 1001],
        "DeviceInfo": ["Windows", "iOS"]
    })
    
    # Mock load_raw
    pipeline.load_raw = lambda: (df_txn, df_id)
    
    # Run pipeline features extraction
    preprocessed_df = pipeline.preprocess((df_txn, df_id))
    final_df = pipeline.extract_features(preprocessed_df)
    
    assert "card1_count_cumulative" in final_df.columns
    assert "amount_to_mean_ratio" in final_df.columns
    
    # Run graph construction
    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_inductive_graph(final_df)
    
    # Nodes: 2 unique cards (101, 102)
    assert graph_data.num_nodes == 2
    # Node features: 339 Vesta features + 4 baseline numerical features + 2 engineered
    assert graph_data.x.shape == (2, 345)
    assert graph_data.y.shape == (2,)

