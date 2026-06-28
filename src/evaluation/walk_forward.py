import logging
from typing import Generator, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Temporal Validation Split Orchestrator.
    Prevents temporal data leakage by partition splits based on time steps.
    """

    def __init__(self, n_splits: int = 5, gap_steps: int = 0):
        self.n_splits = n_splits
        self.gap_steps = gap_steps

    def split(
        self, data: pd.DataFrame, time_col: str = "step"
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generates indices for rolling walk-forward validation.
        Each fold represents a chronologically separated train and validation set.
        """
        logger.info(
            f"Setting up {self.n_splits}-fold temporal walk-forward split on column '{time_col}'..."
        )

        unique_times = sorted(data[time_col].unique())
        if len(unique_times) < self.n_splits + 1:
            # Fallback to simple indices if too few time steps
            logger.warning(
                "Fewer timesteps than requested splits. Splitting based on index ranges."
            )
            idx_split = np.array_split(data.index, self.n_splits + 1)
            for i in range(1, self.n_splits + 1):
                train_idx = np.concatenate(idx_split[:i])
                val_idx = idx_split[i]
                yield data.iloc[train_idx], data.iloc[val_idx]
            return

        # Temporal boundaries
        fold_size = len(unique_times) // (self.n_splits + 1)

        for fold in range(1, self.n_splits + 1):
            train_boundary = unique_times[fold * fold_size]
            val_boundary = unique_times[
                min((fold + 1) * fold_size, len(unique_times) - 1)
            ]

            train_df = data[data[time_col] < train_boundary]
            val_df = data[
                (data[time_col] >= train_boundary + self.gap_steps)
                & (data[time_col] <= val_boundary)
            ]

            logger.info(
                f"Fold {fold}: Train steps < {train_boundary}, Val steps [{train_boundary + self.gap_steps} to {val_boundary}]"
            )
            yield train_df, val_df
