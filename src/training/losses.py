import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Class-weighted Focal Loss for extreme class imbalance in fraud detection.

    Combines two complementary imbalance correction mechanisms:
      1. **Focal modulation** — (1 - p_t)^gamma down-weights easy legitimate
         transactions so the model focuses gradient budget on hard fraud cases.
      2. **Alpha class weighting** — a per-class scalar that can be auto-computed
         from dataset class frequencies rather than fixed at a heuristic value.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Weight for the positive (fraud) class. If None, must be provided
               at construction time via `from_class_counts`. Defaults to 0.75
               which gives fraud 3x gradient priority over legitimate examples.
        gamma: Focusing exponent. Higher values suppress well-classified negatives
               more aggressively. gamma=2 is the standard recommendation.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    @classmethod
    def from_class_counts(
        cls,
        n_negative: int,
        n_positive: int,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> "FocalLoss":
        """
        Factory method: auto-compute alpha from dataset class frequencies.

        Alpha is set to the proportion of negative (majority) samples so that
        the minority (fraud) class receives proportionally higher weight.

        Example: 3% fraud rate -> alpha = 0.97 (fraud gets 97% weight)
        """
        total = n_negative + n_positive
        # Inverse-frequency weighting: fraud class gets weight proportional
        # to how rare it is in the training distribution
        alpha = n_negative / total
        return cls(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Model logits — shape [N, num_classes]
            targets: True class labels — shape [N], dtype torch.long
        Returns:
            Scalar focal loss value (or per-sample tensor if reduction='none')
        """
        # Standard cross-entropy loss (unreduced) gives us -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t: probability assigned to the correct class
        pt = torch.exp(-ce_loss)

        # Per-sample alpha: fraud class gets self.alpha, legitimate gets (1 - self.alpha)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # Focal term: (1-p_t)^gamma — approaches 0 for confident correct predictions
        focal_weight = (1.0 - pt) ** self.gamma

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassWeightedCrossEntropy(nn.Module):
    """
    Standard cross-entropy with inverse-frequency class weighting.

    Simpler alternative to FocalLoss when the dataset imbalance is moderate.
    Useful as a warm-up loss before switching to full focal training.

    Args:
        class_weights: Tensor of per-class weights [n_classes], typically
                       computed as total_samples / (n_classes * class_count).
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    @classmethod
    def from_class_counts(cls, n_negative: int, n_positive: int) -> "ClassWeightedCrossEntropy":
        """
        Factory method: compute inverse-frequency weights from class counts.

        Produces weight tensor [w_negative, w_positive] where each class
        weight = total / (n_classes * class_count).
        """
        total = n_negative + n_positive
        w_neg = total / (2.0 * n_negative)
        w_pos = total / (2.0 * n_positive)
        weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
        return cls(class_weights=weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


class AdversarialGeneratorLoss(nn.Module):
    """
    Loss for the LSTM generator in the adversarial training loop.

    The generator's objective is to synthesize transactions that the GNN
    discriminator classifies as legitimate (class 0). This loss directly
    maximises the probability the discriminator is fooled.
    """

    def __init__(self):
        super().__init__()

    def forward(self, discriminator_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discriminator_logits: GNN output logits for fake transactions [N, 2]
        Returns:
            Scalar loss — minimized when GNN predicts class 0 (legitimate)
        """
        # Target: all generated samples should be classified as class 0 (licit)
        targets = torch.zeros(
            discriminator_logits.size(0),
            dtype=torch.long,
            device=discriminator_logits.device,
        )
        return F.cross_entropy(discriminator_logits, targets)
