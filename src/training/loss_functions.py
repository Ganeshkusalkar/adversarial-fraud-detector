"""
src/training/loss_functions.py
================================
Legacy compatibility shim — imports re-exported from the canonical losses.py.
Kept for backward compatibility with any existing import paths.
Use `from src.training.losses import FocalLoss` in new code.
"""

from src.training.losses import (
    FocalLoss,
    ClassWeightedCrossEntropy,
    AdversarialGeneratorLoss,
)

# Backward-compatible alias
AdversarialLoss = AdversarialGeneratorLoss

__all__ = [
    "FocalLoss",
    "ClassWeightedCrossEntropy",
    "AdversarialLoss",
    "AdversarialGeneratorLoss",
]
