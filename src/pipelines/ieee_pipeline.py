import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from src.pipelines.base_loader import BaseDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("IEEECISPipeline")


class IEEECISPipeline(BaseDataLoader):
    """
    Ingestion and preprocessing pipeline for IEEE-CIS transaction data.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ieee_config = config["data"]["ieee"]
        self.categorical_cols = config["features"]["ieee_categorical"]
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tx_path = self.ieee_config["transaction_path"]
        id_path = self.ieee_config["identity_path"]

        logger.info(f"Loading Transaction data from: {tx_path}")
        df_txn = pd.read_csv(tx_path)

        logger.info(f"Loading Identity data from: {id_path}")
        df_id = pd.read_csv(id_path)

        return df_txn, df_id

    def preprocess(
        self, data: Tuple[pd.DataFrame, pd.DataFrame], fit: bool = True
    ) -> pd.DataFrame:
        """
        Merges transaction and identity tables, handles missing values, and
        encodes categorical columns.
        """
        df_txn, df_id = data

        df = pd.merge(df_txn, df_id, on="TransactionID", how="left")
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        df = df.copy()

        # V-feature Imputation & Missingness Indicators
        v_cols = [c for c in df.columns if c.startswith("V")]
        indicator_dict: Dict[str, pd.Series] = {}
        v_medians = df[v_cols].median()

        for col in v_cols:
            null_mask = df[col].isnull()
            if null_mask.any():
                indicator_dict[f"{col}_is_missing"] = null_mask.astype(np.int8)
                df[col] = df[col].fillna(v_medians[col])

        if indicator_dict:
            indicator_df = pd.DataFrame(indicator_dict, index=df.index)
            df = pd.concat([df, indicator_df], axis=1)

        # Categorical Encoding
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].map(
                            lambda s: s if s in le.classes_ else "UNKNOWN"
                        )
                        if "UNKNOWN" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "UNKNOWN")
                        df[col] = le.transform(df[col])
                    else:
                        df[col] = 0

        # Global Numeric Imputation
        num_cols = df.select_dtypes(include=[np.number]).columns
        col_medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(col_medians)

        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers transaction count and ratio features per card entity.
        """
        new_features: Dict[str, pd.Series] = {}

        # Expanding transaction count per card (shifted to avoid look-ahead bias)
        new_features["card1_count_cumulative"] = df.groupby("card1")[
            "TransactionDT"
        ].transform(lambda x: x.expanding().count().shift(1).fillna(0))

        # Transaction amount relative to card's historical mean spend
        card_mean_amt = df.groupby("card1")["TransactionAmt"].transform("mean")
        new_features["amount_to_mean_ratio"] = df["TransactionAmt"] / (
            card_mean_amt + 1e-5
        )

        engineered_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, engineered_df], axis=1)
        return df

    def run_pipeline(self, fit: bool = True) -> pd.DataFrame:
        raw_data = self.load_raw()
        preprocessed_df = self.preprocess(raw_data, fit=fit)
        final_featured_df = self.extract_features(preprocessed_df)
        return final_featured_df
