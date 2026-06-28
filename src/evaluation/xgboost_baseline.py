import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from src.utils.logger import setup_logger

logger = setup_logger("XGBoostBaseline")

def train_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Trains an XGBoost baseline model on tabular transaction data, completely ignoring graph structure.
    Used to demonstrate the value-add of the GNN.
    """
    logger.info("Initializing XGBoost tabular baseline (No graph structure)...")
    
    # Severe class imbalance handled via scale_pos_weight
    pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False
    )
    
    logger.info("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )
    
    logger.info("Scoring XGBoost on Test Set...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    logger.info(f"XGBoost Test AUC: {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, preds)}")
    
    return model, auc

if __name__ == "__main__":
    # Example execution script for the baseline
    logger.info("XGBoost Baseline Script")
    # Provide dummy data just to ensure it runs
    dummy_X_train = np.random.randn(1000, 345)
    dummy_y_train = np.random.randint(0, 2, 1000)
    dummy_X_test = np.random.randn(200, 345)
    dummy_y_test = np.random.randint(0, 2, 200)
    
    train_baseline(dummy_X_train, dummy_y_train, dummy_X_test, dummy_y_test)
