import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple
from torch_geometric.data import Data
from src.pipelines.base_loader import BaseDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("EllipticPipeline")


class EllipticPipeline(BaseDataLoader):
    """
    Elliptic Bitcoin dataset ingestion pipeline.

    Maps raw transaction nodes and connection edge lists into a PyTorch
    Geometric Data object for GraphSAGE discriminator ingestion. Constructs
    time-based train/val/test masks using a 70/15/15 split over timesteps
    so evaluation reflects realistic temporal generalization.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.elliptic_config = config["data"]["elliptic"]

    # ------------------------------------------------------------------
    # STAGE 1: Raw I/O
    # ------------------------------------------------------------------
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads features, classes, and edge-list from raw Elliptic CSV files."""
        features_path = self.elliptic_config["features_path"]
        classes_path = self.elliptic_config["classes_path"]
        edges_path = self.elliptic_config["edges_path"]

        logger.info(f"Loading Elliptic node features from: {features_path}")
        # No header row — columns are: txId, timestep, f1..f165
        features_df = pd.read_csv(features_path, header=None)

        logger.info(f"Loading Elliptic class labels from: {classes_path}")
        classes_df = pd.read_csv(classes_path)

        logger.info(f"Loading Elliptic edge list from: {edges_path}")
        edges_df = pd.read_csv(edges_path)

        return features_df, classes_df, edges_df

    # ------------------------------------------------------------------
    # STAGE 2: Preprocessing
    # ------------------------------------------------------------------
    def preprocess(
        self, raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Maps class labels and node IDs to sequential integer indices.
        Uses explicit dtype casting to prevent pandas deprecation warnings
        from bare .astype(int) on mixed-type DataFrames.
        """
        features_df, classes_df, edges_df = raw_data

        # Map class labels:  '1' -> 1 (illicit/fraud),  '2' -> 0 (licit),  'unknown' -> -1
        classes_df = classes_df.copy()
        classes_df["label"] = (
            classes_df["class"]
            .map({"1": 1, "2": 0, "unknown": -1})
            .fillna(-1)
            .astype(np.int64)
        )

        # Build txId -> sequential index mapping from the feature node set
        node_ids = features_df.iloc[:, 0].values
        node_map: Dict[int, int] = {
            int(tx_id): idx for idx, tx_id in enumerate(node_ids)
        }

        # Remap edge endpoints to sequential integer indices
        edges_df = edges_df.copy()
        edges_df["source_idx"] = edges_df["txId1"].map(node_map)
        edges_df["target_idx"] = edges_df["txId2"].map(node_map)

        # Drop edges referencing nodes that are absent from the features table
        edges_df = edges_df.dropna(subset=["source_idx", "target_idx"])

        # Explicit int64 casting avoids FutureWarning from bare .astype(int)
        edges_df = edges_df.astype({"source_idx": "int64", "target_idx": "int64"})

        return features_df, classes_df, edges_df

    # ------------------------------------------------------------------
    # STAGE 3: Graph Construction
    # ------------------------------------------------------------------
    def extract_features(
        self, preprocessed_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> Data:
        """
        Constructs a PyTorch Geometric Data object with node features,
        directed edges, labels, and time-based split masks.

        Mask strategy (labeled nodes only):
          - train_mask  : earliest 70% of timesteps
          - val_mask    : next 15% of timesteps
          - test_mask   : final 15% of timesteps (unseen future)
        """
        features_df, classes_df, edges_df = preprocessed_data

        # Node feature matrix — skip txId (col 0) and timestep (col 1)
        x = torch.tensor(features_df.iloc[:, 2:].values, dtype=torch.float32)

        # Node labels tensor
        y = torch.tensor(classes_df["label"].values, dtype=torch.long)

        # Edge index tensor — shape [2, num_edges]
        src_nodes = edges_df["source_idx"].values
        dst_nodes = edges_df["target_idx"].values
        edge_index = torch.tensor(
            np.stack([src_nodes, dst_nodes], axis=0), dtype=torch.long
        )

        # Build the base PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # --- Time-based train/val/test masks ---
        # Column 1 of the raw features_df is the timestep (1..49)
        timesteps = features_df.iloc[:, 1].values.astype(np.int64)
        unique_times = np.unique(timesteps)
        n_times = len(unique_times)

        train_cutoff = unique_times[int(n_times * 0.70)]
        val_cutoff = unique_times[int(n_times * 0.85)]

        labeled_mask = y != -1
        data.train_mask = labeled_mask & torch.tensor(
            timesteps <= train_cutoff, dtype=torch.bool
        )
        data.val_mask = labeled_mask & torch.tensor(
            (timesteps > train_cutoff) & (timesteps <= val_cutoff), dtype=torch.bool
        )
        data.test_mask = labeled_mask & torch.tensor(
            timesteps > val_cutoff, dtype=torch.bool
        )

        logger.info(
            f"Elliptic graph built — Nodes: {data.num_nodes} | Edges: {data.num_edges} | "
            f"Train: {data.train_mask.sum().item()} | "
            f"Val: {data.val_mask.sum().item()} | "
            f"Test: {data.test_mask.sum().item()}"
        )
        return data

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self) -> Data:
        """Orchestrates the complete Elliptic dataset processing pipeline."""
        raw_data = self.load_raw()
        preprocessed = self.preprocess(raw_data)
        graph_data = self.extract_features(preprocessed)
        return graph_data
