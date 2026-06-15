import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.models.discriminator_gnn import FraudGNN
from src.models.generator_lstm import FraudTransactionGenerator
from src.training.losses import FocalLoss, AdversarialGeneratorLoss
from src.utils.logger import setup_logger

logger = setup_logger("TrainingEngine")


class AdversarialTrainingEngine:
    """
    Orchestrates the min-max adversarial training loop between:
      - The **Discriminator** (FraudGNN):  tries to correctly flag fraud
      - The **Generator** (FraudTransactionGenerator): tries to fool the GNN

    Production-grade hardening applied:
      - Focal Loss with auto-computed alpha for class imbalance correction
      - CosineAnnealingLR on the discriminator for smooth convergence
      - Gradient clipping on both networks to prevent exploding gradients
      - Best-epoch checkpoint saving based on discriminator loss
    """

    def __init__(
        self,
        gnn_model: FraudGNN,
        gen_model: FraudTransactionGenerator,
        device: str = "cpu",
        n_negative: int = 95000,
        n_positive: int = 5000,
        checkpoint_dir: str = "artifacts/checkpoints",
        total_epochs: int = 30,
        sample_weights = None,
    ):
        """
        Args:
            gnn_model:       Instantiated FraudGNN discriminator.
            gen_model:       Instantiated FraudTransactionGenerator.
            device:          Compute device ('cpu' or 'cuda').
            n_negative:      Count of legitimate samples in the training graph.
                             Used to auto-compute focal loss alpha.
            n_positive:      Count of fraudulent samples in the training graph.
            checkpoint_dir:  Directory to write best-model checkpoints.
            total_epochs:    Total training epochs (used by LR scheduler).
        """
        self.gnn = gnn_model.to(device)
        self.generator = gen_model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.sample_weights = sample_weights
        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- Optimizers (FAANG-standard Adam with tuned learning rates) ---
        self.opt_D = torch.optim.Adam(self.gnn.parameters(), lr=1e-3, weight_decay=1e-4)
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)

        # --- Learning Rate Scheduler (smooth cosine decay for discriminator) ---
        # CosineAnnealingLR prevents loss spikes from fixed LR in later epochs
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D, T_max=total_epochs, eta_min=1e-5
        )

        # --- Loss Functions ---
        # Focal Loss: auto-alpha derived from class distribution for maximum recall
        self.criterion_D = FocalLoss.from_class_counts(
            n_negative=n_negative,
            n_positive=n_positive,
            gamma=2.0,
        )
        logger.info(
            f"FocalLoss initialized with auto-alpha={self.criterion_D.alpha:.4f} "
            f"(fraud weight ∝ {n_negative / n_positive:.1f}x class imbalance ratio)"
        )

        # Generator loss: adversarial objective — fool discriminator into predicting legitimate
        self.criterion_G = AdversarialGeneratorLoss()

        # --- Gradient clipping threshold (prevents NaN losses in GAN loops) ---
        self.grad_clip_norm = 1.0

        # --- Best checkpoint tracking ---
        self.best_disc_loss = float("inf")

    # ------------------------------------------------------------------
    # Single Epoch Training
    # ------------------------------------------------------------------
    def train_epoch(self, real_graph_data: Data, batch_size: int = 64) -> dict:
        """
        Executes one complete adversarial training epoch.

        Returns:
            dict with keys: discriminator_loss, generator_loss
        """
        real_graph_data = real_graph_data.to(self.device)


        # ----------------------------------------------------------------
        # STEP 1: Train Discriminator (GNN Defender)
        # Objective: Correctly classify real fraud AND flag synthetic fakes
        # ----------------------------------------------------------------
        self.gnn.train()
        self.generator.eval()  # Freeze generator BN/dropout during discriminator step
        self.opt_D.zero_grad()

        # 1a. Focal loss on the real labeled graph nodes
        real_preds = self.gnn(real_graph_data.x, real_graph_data.edge_index)
        
        if self.sample_weights is not None:
            # Sample a number of nodes equal to the full graph size to maintain epoch scale
            num_samples_to_draw = len(self.sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(self.sample_weights, num_samples_to_draw, replacement=True)
            sampled_indices = list(sampler)
            loss_real = self.criterion_D(real_preds[sampled_indices], real_graph_data.y[sampled_indices])
        else:
            loss_real = self.criterion_D(real_preds, real_graph_data.y)

        # 1b. Generate adversarial synthetic transactions (detached — no G gradients)
        noise = self.generator.sample_noise(batch_size, device=self.device)
        synthetic_features = self.generator(noise).detach()  # [B, seq_len, feat_dim]

        # Flatten to first-timestep features and pad to match GNN input dimensionality
        fake_flat = synthetic_features.mean(dim=1)  # [B, feat_dim]

        # Self-loop edges for the mock synthetic batch (no graph structure — node-wise scoring)
        mock_edges = torch.stack([torch.arange(batch_size), torch.arange(batch_size)]).to(self.device)

        fake_preds = self.gnn(fake_flat, mock_edges)
        # Tell the GNN these ARE fraud (class 1) — train it to catch adversarial inputs
        fake_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss_fake = self.criterion_D(fake_preds, fake_labels)

        # Combined discriminator loss: real data is primary, fake data auxiliary (0.5x)
        loss_D = loss_real + 0.5 * loss_fake
        loss_D.backward()

        # Gradient clipping prevents GAN instability from large gradient steps
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.grad_clip_norm)
        self.opt_D.step()

        # ----------------------------------------------------------------
        # STEP 2: Train Generator (Attacker Agent)
        # Objective: Synthesize transactions that the GNN classifies as LICIT
        # ----------------------------------------------------------------
        self.gnn.eval()  # Freeze discriminator weights during generator update
        self.generator.train()
        self.opt_G.zero_grad()

        # Re-sample fresh noise and re-generate — don't reuse detached samples
        noise = self.generator.sample_noise(batch_size, device=self.device)
        synthetic_features = self.generator(noise)
        fake_flat = synthetic_features.mean(dim=1)

        # Run through frozen discriminator
        fool_preds = self.gnn(fake_flat, mock_edges)

        # Generator's adversarial loss: push GNN to predict class 0 (legitimate)
        loss_G = self.criterion_G(fool_preds)
        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_norm)
        self.opt_G.step()

        return {
            "discriminator_loss": loss_D.item(),
            "generator_loss": loss_G.item(),
        }

    # ------------------------------------------------------------------
    # Full Training Orchestration
    # ------------------------------------------------------------------
    def run_training_orchestration(
        self, real_graph_data: Data, total_epochs: int = 30, batch_size: int = 64
    ) -> dict:
        """
        Runs the full adversarial training loop, stepping the LR scheduler,
        and saving the best discriminator checkpoint after each epoch.

        Returns:
            dict with best_epoch and best_discriminator_loss.
        """
        logger.info(
            f"Starting adversarial training for {total_epochs} epochs | "
            f"Batch size: {batch_size} | Device: {self.device}"
        )

        best_epoch = 1
        epoch_history = []

        for epoch in range(1, total_epochs + 1):
            metrics = self.train_epoch(real_graph_data, batch_size)
            epoch_history.append(metrics)

            # Step LR scheduler after each epoch
            self.scheduler_D.step()

            # Save best discriminator checkpoint
            if metrics["discriminator_loss"] < self.best_disc_loss:
                self.best_disc_loss = metrics["discriminator_loss"]
                best_epoch = epoch
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_gnn.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.gnn.state_dict(),
                        "optimizer_state_dict": self.opt_D.state_dict(),
                        "discriminator_loss": self.best_disc_loss,
                    },
                    checkpoint_path,
                )

            if epoch % 5 == 0 or epoch == 1:
                current_lr = self.scheduler_D.get_last_lr()[0]
                logger.info(
                    f"Epoch [{epoch:3d}/{total_epochs}] | "
                    f"D Loss: {metrics['discriminator_loss']:.4f} | "
                    f"G Loss: {metrics['generator_loss']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Best: Epoch {best_epoch} ({self.best_disc_loss:.4f})"
                )

        logger.info(
            f"Adversarial hardening complete. "
            f"Best discriminator checkpoint at epoch {best_epoch} "
            f"(loss={self.best_disc_loss:.4f}) saved to {self.checkpoint_dir}/best_gnn.pt"
        )

        return {
            "best_epoch": best_epoch,
            "best_discriminator_loss": self.best_disc_loss,
            "history": epoch_history,
        }

# Bugfix: corrected mock edges and generator sequence pooling in gan loop
