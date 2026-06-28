import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from src.utils.logger import setup_logger

logger = setup_logger("EvaluationMetrics")


class FraudEvaluationEngine:
    def __init__(self, config: dict):
        self.config = config
        self.target_precision_k = 0.01  # Top 1% of highest-risk flagged transactions
        # Read the tuned decision threshold — must match training evaluation threshold
        self.decision_threshold = config.get("training", {}).get(
            "decision_threshold", 0.38
        )

    def compute_standard_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """
        Computes the standard suite of industry-standard fraud detection metrics.

        Binary classification metrics (Precision, Recall, F1) are evaluated at
        the configurable decision_threshold from base_config.yaml rather than
        the naive 0.5 default, so results are consistent with training evaluation
        and production inference.

        Threshold-independent metrics (AUC-ROC, Average Precision, Precision@K)
        are always computed over the full probability distribution.
        """
        # Use the tuned threshold from config (default 0.38) — not hard-coded 0.5
        # This ensures Recall/Precision/F1 match what training and test_predictions report
        y_pred = (y_prob >= self.decision_threshold).astype(int)

        auc_roc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        # Isolate scores specifically for the positive class (1 = Fraud)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Calculate Precision at K (Top 1% highest-score elements)
        k_elements = int(len(y_prob) * self.target_precision_k)
        if k_elements > 0:
            top_k_indices = np.argsort(y_prob)[-k_elements:]
            precision_at_k = np.mean(y_true[top_k_indices])
        else:
            precision_at_k = 0.0

        metrics = {
            "AUC-ROC": float(auc_roc),
            "Average_Precision": float(avg_precision),
            "Precision_Fraud_Class": float(precision),
            "Recall_Sensitivity": float(recall),
            "F1_Score_Fraud_Class": float(f1),
            "Precision_at_Top_1_Percent": float(precision_at_k),
        }

        logger.info("=== Operational Performance Diagnostics ===")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        return metrics

    def calculate_adversarial_robustness(
        self,
        hardened_model: torch.nn.Module,
        generator: torch.nn.Module,
        n_attacks: int = 1000,
        device: str = "cpu",
    ) -> float:
        """
        Simulates an advanced attacker utilizing the trained generator network to mount
        highly disguised evasion attacks against the final production model.
        """
        hardened_model.eval()
        generator.eval()

        caught_attacks = 0
        pad_size = hardened_model.conv1.in_channels - generator.feature_dim
        mock_edges = torch.zeros((2, 1), dtype=torch.long, device=device)

        logger.info(
            f"Simulating {n_attacks} automated adversarial mutation stress tests against the GNN..."
        )

        with torch.no_grad():
            for _ in range(n_attacks):
                # Sample random latent noise and generate the optimal disguised attack path
                noise = generator.sample_noise(1, device=device)
                fake_txn = generator(noise)  # Shape: (1, seq_len, feature_dim)

                # Format to match GNN spatial input shapes
                padded_features = torch.cat(
                    [fake_txn[:, 0, :], torch.zeros((1, pad_size), device=device)],
                    dim=-1,
                )

                # Run inference through the hardened model
                prediction_logits = hardened_model(padded_features, mock_edges)
                predicted_class = prediction_logits.argmax(dim=-1).item()

                # If the defender correctly flags it as class 1 (Fraud), increment tracking
                if predicted_class == 1:
                    caught_attacks += 1

        robustness_score = caught_attacks / n_attacks
        logger.info(
            f"Adversarial Robustness Score: {robustness_score:.3f} ({caught_attacks}/{n_attacks} attacks intercepted)"
        )
        return robustness_score

    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Computes the full Precision-Recall curve to find the threshold that maximizes F1.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # PR curve returns arrays where len(p) == len(r) == len(t) + 1
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Log top 10 candidate thresholds
        top_10_idx = np.argsort(f1_scores)[-10:][::-1]
        logger.info("Top 10 Thresholds for F1-Score:")
        for idx in top_10_idx:
            logger.info(
                f"  Threshold: {thresholds[idx]:.4f} | Precision: {precision[idx]:.4f} | Recall: {recall[idx]:.4f} | F1: {f1_scores[idx]:.4f}"
            )

        # Log threshold that achieves exactly ~90% Recall (business floor)
        valid_recall_idx = np.where(recall >= 0.90)[0]
        if len(valid_recall_idx) > 0:
            best_90_idx = valid_recall_idx[
                -1
            ]  # thresholds are increasing, recall is decreasing
            logger.info(
                f"Threshold for >= 90% Recall: {thresholds[best_90_idx]:.4f} (Recall: {recall[best_90_idx]:.4f}, Precision: {precision[best_90_idx]:.4f})"
            )

        return float(optimal_threshold)

    def find_threshold_with_constraints(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_recall: float = 0.85,
        min_precision: float = 0.40,
    ) -> float:
        """
        Finds the threshold that maximizes F1 subject to Precision/Recall constraints.
        Relaxes precision floor if no threshold satisfies both.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

        current_min_precision = min_precision
        while current_min_precision >= 0.0:
            valid_idx = np.where(
                (recall >= min_recall) & (precision >= current_min_precision)
            )[0]
            if len(valid_idx) > 0:
                valid_f1s = f1_scores[valid_idx]
                best_valid_idx = valid_idx[np.argmax(valid_f1s)]
                best_thresh = float(thresholds[best_valid_idx])
                logger.info(
                    f"Found threshold {best_thresh:.4f} satisfying Recall >= {min_recall} and Precision >= {current_min_precision:.2f}"
                )
                return best_thresh
            else:
                logger.warning(
                    f"No threshold satisfies Recall >= {min_recall} and Precision >= {current_min_precision:.2f}. Relaxing precision by 0.05..."
                )
                current_min_precision -= 0.05

        logger.warning(
            "Failed to find any constrained threshold. Falling back to global optimal F1 threshold."
        )
        return self.find_optimal_threshold(y_true, y_prob)
