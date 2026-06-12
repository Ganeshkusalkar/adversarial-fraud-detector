import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from src.pipelines.base_loader import BaseDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("PaySimPipeline")


class PaySimPipeline(BaseDataLoader):
    """
    PaySim mobile money transaction sequence pipeline.

    Filters the synthetic mobile-money log to fraud-relevant transaction
    types, engineers real-time balance audit signals, then groups account
    histories into fixed-length temporal sequences for LSTM ingestion.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.paysim_config = config["data"]["paysim"]
        self.feature_cols: List[str] = config["features"]["paysim_sequence_features"]
        self.sequence_length: int = config["model"]["generator"]["sequence_length"]

    # ------------------------------------------------------------------
    # STAGE 1: Raw I/O
    # ------------------------------------------------------------------
    def load_raw(self) -> pd.DataFrame:
        """Loads the PaySim transaction log from the configured CSV path."""
        path = self.paysim_config["path"]
        logger.info(f"Loading raw PaySim transaction log from: {path}")
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # STAGE 2: Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters to fraud-relevant transaction types and engineers three
        balance consistency audit signals used as fraud discriminators.
        """
        # Input validation — ensure all required feature columns will exist after engineering
        required_raw = {"amount", "newbalanceOrig", "oldbalanceOrig", "newbalanceDest", "oldbalanceDest", "isFraud"}
        missing = required_raw - set(df.columns)
        if missing:
            raise ValueError(
                f"PaySim dataset is missing required columns: {missing}. "
                "Ensure you are loading the correct PaySim CSV file."
            )

        logger.info("Filtering to TRANSFER and CASH_OUT types (fraud-exclusive channels)...")
        # Optimization: PaySim fraud is strictly confined to these two transaction types
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
        df = df.sort_values("step").reset_index(drop=True)

        logger.info("Engineering real-time balance audit signals...")

        # Signal 1: Sender balance change should equal the deducted amount exactly
        # Non-zero deviation implies a disguised or split transaction
        df["orig_balance_diff"] = (
            df["newbalanceOrig"] - df["oldbalanceOrig"] + df["amount"]
        ).astype(np.float32)

        # Signal 2: Receiver balance increase should match the sent amount exactly
        # Shortfall indicates layering — funds diverted before credited
        df["dest_balance_diff"] = (
            df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
        ).astype(np.float32)

        # Signal 3: Binary flag for accounts intentionally drained to zero
        # A fully zeroed origin account is a classic cash-out fraud signature
        df["orig_zeroed_out"] = (df["newbalanceOrig"] == 0).astype(np.int8)

        # Validate all engineered feature columns are now present
        missing_features = [c for c in self.feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Feature engineering failed — the following configured sequence "
                f"features are missing from the processed DataFrame: {missing_features}"
            )

        return df

    # ------------------------------------------------------------------
    # STAGE 3: Sequence Construction
    # ------------------------------------------------------------------
    def extract_features(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, int]]:
        """
        Groups transactions chronologically by origin account and chunks
        each account's history into overlapping fixed-length windows ready
        for LSTM sequence modelling.

        Returns:
            List of (feature_sequence, label) tuples where:
              - feature_sequence: float32 array of shape (sequence_length, n_features)
              - label: 1 if any step in the window was flagged as fraud, else 0
        """
        logger.info("Constructing per-account temporal sequence chunks...")
        sequences: List[Tuple[np.ndarray, int]] = []

        for acct, grp in df.groupby("nameOrig"):
            if len(grp) < self.sequence_length:
                continue  # Not enough history for a full window — skip

            # Ensure strict chronological ordering by step (hour index)
            sorted_grp = grp.sort_values("step")
            feature_matrix = sorted_grp[self.feature_cols].values.astype(np.float32)

            # Account-level fraud label: fraudulent if ANY window step was flagged
            label = int(sorted_grp["isFraud"].max())

            # Slide a fixed-length window across the full account history
            for i in range(len(feature_matrix) - self.sequence_length + 1):
                seq_chunk = feature_matrix[i : i + self.sequence_length]
                sequences.append((seq_chunk, label))

        logger.info(
            f"PaySim sequence extraction complete — "
            f"Total sequences: {len(sequences):,} | "
            f"Sequence shape: ({self.sequence_length}, {len(self.feature_cols)})"
        )
        return sequences

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self) -> List[Tuple[np.ndarray, int]]:
        """Orchestrates the complete PaySim data preparation pipeline."""
        raw_df = self.load_raw()
        processed_df = self.preprocess(raw_df)
        final_sequences = self.extract_features(processed_df)
        return final_sequences
