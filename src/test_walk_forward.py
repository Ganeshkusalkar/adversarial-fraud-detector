"""
src/test_walk_forward.py
========================
Executes temporal Walk-Forward Validation (Test 1 from requirements).
Proves that the model generalizes to future, unseen time periods
without temporal data leakage (unlike naive random train/test splits).
"""

import pandas as pd
import numpy as np
import yaml
import torch
import copy
from sklearn.preprocessing import StandardScaler

from src.pipelines.ieee_pipeline import IEEECISPipeline
from src.graph.graph_builder import TransactionGraphBuilder
from src.evaluation.walk_forward import WalkForwardValidator
from src.evaluation.metrics import FraudEvaluationEngine
from src.models.discriminator_gnn import FraudGNN
from src.utils.logger import setup_logger

logger = setup_logger("WalkForwardTest")


def run_walk_forward_test():
    logger.info("=" * 70)
    logger.info("  TEMPORAL WALK-FORWARD VALIDATION (TEST 1)")
    logger.info("=" * 70)

    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tx_path = config["data"]["ieee"]["transaction_path"]
    id_path = config["data"]["ieee"]["identity_path"]

    # Load a temporal subset
    logger.info("Loading chronological data for temporal splits...")
    df_txn = pd.read_csv(tx_path, nrows=30_000).sort_values("TransactionDT")
    df_id = pd.read_csv(id_path)

    pipeline = IEEECISPipeline(config)
    pipeline.load_raw = lambda: (df_txn, df_id)
    processed_df = pipeline.run_pipeline()

    # 5-fold walk forward
    validator = WalkForwardValidator(n_splits=3, gap_steps=0)

    fold_aucs = []

    for fold_idx, (train_df, val_df) in enumerate(
        validator.split(processed_df, time_col="TransactionDT"), 1
    ):
        logger.info(f"\n--- Running Fold {fold_idx}/3 ---")
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        # Build graphs independently to prevent structural leakage
        builder = TransactionGraphBuilder(config)
        train_graph = builder.build_inductive_graph(train_df)
        val_graph = builder.build_inductive_graph(val_df)

        # Scale
        scaler = StandardScaler()
        train_graph.x = torch.tensor(
            scaler.fit_transform(train_graph.x.numpy()), dtype=torch.float32
        )
        val_graph.x = torch.tensor(
            scaler.transform(val_graph.x.numpy()), dtype=torch.float32
        )

        # Initialize GNN
        model = FraudGNN(
            in_channels=config["model"]["gnn"]["in_channels"],
            hidden_channels=64,  # smaller for fast test
            out_channels=2,
            dropout=0.2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Quick train (10 epochs)
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            out = model(train_graph.x, train_graph.edge_index)
            loss = criterion(out, train_graph.y)
            loss.backward()
            optimizer.step()

        # Evaluate on strictly future data
        model.eval()
        with torch.no_grad():
            val_out = model(val_graph.x, val_graph.edge_index)
            # Softmax
            exp_l = torch.exp(val_out - val_out.max(dim=1, keepdim=True)[0])
            probs = (exp_l / exp_l.sum(dim=1, keepdim=True))[:, 1].numpy()

        eval_engine = FraudEvaluationEngine(config)
        metrics = eval_engine.compute_standard_metrics(val_graph.y.numpy(), probs)

        auc = metrics["AUC-ROC"]
        logger.info(f"Fold {fold_idx} Out-of-Sample AUC: {auc:.4f}")
        fold_aucs.append(auc)

    avg_auc = np.mean(fold_aucs)
    logger.info("=" * 70)
    logger.info(f"Average Temporal Out-of-Sample AUC: {avg_auc:.4f}")
    if avg_auc >= 0.85:
        logger.info(
            "[PASS] STATUS: PASS - Model generalizes well to future time periods."
        )
    else:
        logger.info("[FAIL] STATUS: FAIL - Model overfits to historical data.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_walk_forward_test()
