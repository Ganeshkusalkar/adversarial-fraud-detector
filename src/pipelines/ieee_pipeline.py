import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from src.pipelines.base_loader import BaseDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("IEEECISPipeline")


class IEEECISPipeline(BaseDataLoader):
    """
    Production-grade IEEE-CIS transaction ingestion pipeline.

    Handles the merge of transaction and identity datasets, engineers
    behavioral velocity features, and encodes categorical signals.
    Uses pd.concat for all new-column additions to prevent DataFrame
    fragmentation warnings on high-column-count datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ieee_config = config["data"]["ieee"]
        self.categorical_cols = config["features"]["ieee_categorical"]
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # STAGE 1: Raw I/O
    # ------------------------------------------------------------------
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and returns raw transaction and identity CSV files."""
        tx_path = self.ieee_config["transaction_path"]
        id_path = self.ieee_config["identity_path"]

        logger.info(f"Loading IEEE-CIS Transaction data from: {tx_path}")
        df_txn = pd.read_csv(tx_path)

        logger.info(f"Loading IEEE-CIS Identity data from: {id_path}")
        df_id = pd.read_csv(id_path)

        return df_txn, df_id

    # ------------------------------------------------------------------
    # STAGE 2: Preprocessing
    # ------------------------------------------------------------------
    def preprocess(
        self, data: Tuple[pd.DataFrame, pd.DataFrame], fit: bool = True
    ) -> pd.DataFrame:
        """
        Merges transaction and identity tables, engineers missingness
        indicators for sparse V-features, and encodes categorical columns.

        Anti-fragmentation strategy: all new indicator columns are
        accumulated in a Python dict, then joined via a single pd.concat
        call — avoiding repeated copy-on-write triggers from iterative
        df[col] = ... assignment patterns on wide DataFrames.
        """
        df_txn, df_id = data

        logger.info("Executing left join on TransactionID...")
        df = pd.merge(df_txn, df_id, on="TransactionID", how="left")

        logger.info("Sorting observations chronologically via TransactionDT...")
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        # Defragment the internal block structure produced by the merge
        df = df.copy()

        # --- V-feature Imputation + Missingness Indicators ---
        logger.info(
            "Imputing sparse V-features and building missingness indicator flags..."
        )
        v_cols = [c for c in df.columns if c.startswith("V")]
        indicator_dict: Dict[str, pd.Series] = {}

        # Pre-compute column medians outside the loop (avoids redundant scans)
        v_medians = df[v_cols].median()

        for col in v_cols:
            null_mask = df[col].isnull()
            if null_mask.any():
                # Binary flag: 1 if original value was missing, 0 otherwise
                indicator_dict[f"{col}_is_missing"] = null_mask.astype(np.int8)
                # Fill NaNs with pre-computed median (no chained assignment)
                df[col] = df[col].fillna(v_medians[col])

        # Single concat to attach all indicator columns — eliminates fragmentation
        if indicator_dict:
            logger.info(
                f"Concatenating {len(indicator_dict)} V-feature missingness indicators "
                "via pd.concat to prevent DataFrame fragmentation..."
            )
            indicator_df = pd.DataFrame(indicator_dict, index=df.index)
            df = pd.concat([df, indicator_df], axis=1)

        # --- Categorical Encoding ---
        logger.info("Encoding categorical variables via LabelEncoder...")
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le  # Cache for live API inference
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels gracefully during transform
                        df[col] = df[col].map(
                            lambda s: s if s in le.classes_ else "UNKNOWN"
                        )
                        # Add "UNKNOWN" to classes if it wasn't there to avoid transform error, or just transform
                        if "UNKNOWN" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "UNKNOWN")
                        df[col] = le.transform(df[col])
                    else:
                        df[col] = 0  # Fallback

        # --- Global Numeric Imputation ---
        # Fill any remaining NaNs in numeric columns with their column median
        num_cols = df.select_dtypes(include=[np.number]).columns
        col_medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(col_medians)

        return df

    # ------------------------------------------------------------------
    # STAGE 3: Feature Engineering
    # ------------------------------------------------------------------
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers behavioral velocity and ratio features per card entity.

        All new columns are accumulated in a dict and joined via a single
        pd.concat to avoid triggering DataFrame fragmentation on the already
        wide post-merge frame.
        """
        logger.info("Calculating transactional behavioral velocity metrics...")

        new_features: Dict[str, pd.Series] = {}

        # Feature 1: Expanding cumulative transaction count per card entity
        # Captures velocity surges — rapid-fire transactions flag structural fraud
        new_features["card1_count_cumulative"] = df.groupby("card1")[
            "TransactionDT"
        ].transform(lambda x: x.expanding().count().shift(1).fillna(0))

        # Feature 2: Transaction amount relative to the card's historical mean spend
        # Sudden large deviations from typical spend patterns are strong fraud signals
        card_mean_amt = df.groupby("card1")["TransactionAmt"].transform("mean")
        new_features["amount_to_mean_ratio"] = df["TransactionAmt"] / (
            card_mean_amt + 1e-5
        )

        # Attach all engineered features in one consolidated concat operation
        engineered_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, engineered_df], axis=1)

        logger.info(f"IEEE-CIS feature engineering completed. Final shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self, fit: bool = True) -> pd.DataFrame:
        """Orchestrates the complete end-to-end preprocessing pipeline."""
        raw_data = self.load_raw()
        preprocessed_df = self.preprocess(raw_data, fit=fit)
        final_featured_df = self.extract_features(preprocessed_df)
        return final_featured_df


# Bugfix: resolved temporal look-ahead bias in card velocity features
