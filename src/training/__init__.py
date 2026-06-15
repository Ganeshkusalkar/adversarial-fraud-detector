"""
Self-Adversarial GAN Training Loops and Loss Configurations
"""

from .engine import AdversarialTrainingEngine
from .losses import FocalLoss

__all__ = ["AdversarialTrainingEngine", "FocalLoss"]
