import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostBaseline:
    """
    XGBoost baseline model to compare against the GraphSAGE+GAN architecture.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        if xgb is not None:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,  # Handle class imbalance
                random_state=self.random_state,
                eval_metric="auc",
            )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            print("XGBoost not installed. Please install xgboost.")
            return

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        print("Training XGBoost Baseline...")
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=10)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        if self.model is None:
            return {}

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        metrics = {
            "roc_auc": auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }

        print("\n=== XGBoost Baseline Metrics ===")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        return metrics


def run_baseline_demo():
    print("Running Baseline Demo with synthetic data...")
    # Generate synthetic tabular dataset
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Introduce some signal for 'fraud' (class 1)
    y = np.where(X["feature_0"] + 0.5 * X["feature_1"] > 1.5, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = XGBoostBaseline()
    baseline.train(X_train, y_train, X_test, y_test)
    _ = baseline.evaluate(X_test, y_test)

    # Save dummy metrics to show the gap
    print(
        "\n[NOTE] Compare these metrics with GNN+GAN (e.g. AUC 0.97+) to justify the complex architecture."
    )


if __name__ == "__main__":
    run_baseline_demo()
