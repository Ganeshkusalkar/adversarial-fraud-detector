import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from loguru import logger
import torch
from torch_geometric.data import HeteroData
from src.models.gnn_models import build_hetero_graph

def preprocess_ieee_cis(ieee_path: str, sample_size: int = 10000) -> pd.DataFrame:
    """
    Ingest and preprocess IEEE-CIS dataset. If raw files are not found,
    generates a synthetic representation for pipeline testing.
    """
    path = Path(ieee_path)
    train_trans_path = path / "train_transaction.csv"
    train_ident_path = path / "train_identity.csv"
    
    if train_trans_path.exists():
        logger.info(f"Loading IEEE-CIS transaction data from {train_trans_path}")
        # Read a sample of the massive file to preserve memory
        df_trans = pd.read_csv(train_trans_path, nrows=sample_size)
        if train_ident_path.exists():
            df_ident = pd.read_csv(train_ident_path, nrows=sample_size)
            df = pd.merge(df_trans, df_ident, on="TransactionID", how="left")
        else:
            df = df_trans
    else:
        logger.warning(f"IEEE-CIS data not found at {path}. Generating synthetic dataset.")
        # Generate synthetic IEEE-like dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "TransactionID": np.arange(10000, 10000 + sample_size),
            "TransactionAmt": np.random.exponential(scale=100.0, size=sample_size),
            "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], size=sample_size),
            "card_id": np.random.randint(100, 200, size=sample_size),
            "addr_id": np.random.randint(2000, 2100, size=sample_size),
            "email_id": np.random.randint(10, 20, size=sample_size),
            "device_id": np.random.randint(50, 60, size=sample_size),
            "ip_id": np.random.randint(500, 550, size=sample_size),
            "merchant_id": np.random.randint(1000, 1050, size=sample_size),
            "dist1": np.random.normal(loc=10.0, scale=5.0, size=sample_size),
            "dist2": np.random.normal(loc=15.0, scale=7.0, size=sample_size),
            "isFraud": np.random.choice([0, 1], p=[0.95, 0.05], size=sample_size)
        })
        
    # Standard imputation and encoding
    df = df.fillna(df.select_dtypes(include=[np.number]).median())
    
    # Categorical encoding
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
        
    return df

def extract_paysim_sequences(paysim_path: str, seq_len: int = 10, feature_dim: int = 30, sample_size: int = 10000):
    """
    Load PaySim transactions and structure them into temporal sequences.
    """
    path = Path(paysim_path)
    csv_path = path / "paysim dataset.csv"
    
    if csv_path.exists():
        logger.info(f"Loading PaySim data from {csv_path}")
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        logger.warning(f"PaySim data not found at {csv_path}. Generating synthetic sequence data.")
        # Create synthetic temporal dataframe
        df = pd.DataFrame({
            "step": np.repeat(np.arange(1, seq_len + 1), sample_size // seq_len),
            "amount": np.random.exponential(scale=500, size=sample_size),
            "oldbalanceOrg": np.random.exponential(scale=2000, size=sample_size),
            "newbalanceOrig": np.random.exponential(scale=1500, size=sample_size),
            "oldbalanceDest": np.random.exponential(scale=3000, size=sample_size),
            "newbalanceDest": np.random.exponential(scale=3500, size=sample_size),
            "isFraud": np.random.choice([0, 1], p=[0.97, 0.03], size=sample_size)
        })
        # Add remaining features to match feature_dim
        for i in range(feature_dim - 6):
            df[f"feat_{i}"] = np.random.normal(size=sample_size)
            
    # Scale features
    scaler = StandardScaler()
    feat_cols = [c for c in df.columns if c not in ["isFraud", "step"]]
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    
    # Group by steps/sequences
    num_seqs = len(df) // seq_len
    x_seqs = []
    y_seqs = []
    
    for i in range(num_seqs):
        chunk = df.iloc[i * seq_len: (i + 1) * seq_len]
        x_seqs.append(chunk[feat_cols].values)
        y_seqs.append(chunk["isFraud"].values[-1]) # Use label of the final transaction
        
    return np.array(x_seqs), np.array(y_seqs)

def build_elliptic_graph(elliptic_path: str, sample_size: int = 10000):
    """
    Loads Elliptic Bitcoin data and creates a subgraph/mock equivalent.
    """
    path = Path(elliptic_path)
    classes_path = path / "elliptic_txs_classes.csv"
    edgelist_path = path / "elliptic_txs_edgelist.csv"
    features_path = path / "elliptic_txs_features.csv"
    
    if classes_path.exists() and edgelist_path.exists() and features_path.exists():
        logger.info(f"Loading Elliptic dataset from {path}")
        # Since Elliptic features file doesn't have headers, read it carefully
        df_features = pd.read_csv(features_path, nrows=sample_size, header=None)
        df_classes = pd.read_csv(classes_path, nrows=sample_size)
        df_edges = pd.read_csv(edgelist_path, nrows=sample_size)
        
        # Parse features and classes
        df_classes["class"] = df_classes["class"].map({"1": 1, "2": 0, "unknown": -1})
        return df_features, df_classes, df_edges
    else:
        logger.warning(f"Elliptic dataset files not found at {path}. Mocking graph outputs.")
        return None

def run_feature_engineering_pipeline(config):
    """
    Main pipeline to execute all feature building steps.
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Preprocess IEEE-CIS
    ieee_df = preprocess_ieee_cis(config.dataset.ieee_path, config.dataset.sample_size)
    
    # Extract PaySim sequences
    seq_x, seq_y = extract_paysim_sequences(
        config.dataset.paysim_path,
        config.model.seq_len,
        config.model.feature_dim,
        config.dataset.sample_size
    )
    
    # Build Heterogeneous PyG graph from IEEE-CIS features + transaction mapping
    # Keep top columns matching feature_dim
    cols_to_keep = ["TransactionAmt", "ProductCD", "dist1", "dist2", "isFraud",
                    "card_id", "addr_id", "email_id", "device_id", "ip_id", "merchant_id"]
    for i in range(config.model.feature_dim - len(cols_to_keep) + 1):
        ieee_df[f"extra_{i}"] = np.random.normal(size=len(ieee_df))
    
    pyg_data = build_hetero_graph(ieee_df)
    
    # Save processed directories
    processed_dir = Path(config.dataset.processed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / "fraud_sequences.npy", seq_x[seq_y == 1])
    np.save(processed_dir / "legit_sequences.npy", seq_x[seq_y == 0])
    
    # Save graph representation
    torch.save(pyg_data, processed_dir / "hetero_graph.pt")
    logger.info(f"Feature engineering pipeline completed. Saved files to {processed_dir}")
    
    return pyg_data, seq_x, seq_y
