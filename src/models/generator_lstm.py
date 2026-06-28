import torch
import torch.nn as nn
from src.utils.logger import setup_logger

logger = setup_logger("GeneratorLSTM")


class FraudTransactionGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dim: int = 128,
        sequence_length: int = 10,
        feature_dim: int = 345,
        num_layers: int = 2,
    ):
        """
        LSTM-based Adversarial Sequence Generator to synthesize realistic transaction flows.

        Args:
            noise_dim: Dimension of the input random Gaussian noise vector.
            hidden_dim: Hidden dimension size of internal LSTM cells.
            sequence_length: Target length of the multi-step transaction sequence.
            feature_dim: Number of continuous transaction features to synthesize.
            num_layers: Depth of the stacked LSTM architecture.
        """
        super().__init__()
        logger.info(
            f"Initializing FraudTransactionGenerator: Noise ({noise_dim}) -> Sequence Features ({feature_dim})"
        )

        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim

        # Stacked LSTM layer to build temporal dependency relationships across sequential steps
        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Linear layer mapping hidden states back to target transaction feature spaces
        self.fc = nn.Linear(hidden_dim, feature_dim)

        # Binds output array strictly into [-1, 1] bounds for processing stability
        self.tanh = nn.Tanh()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Transforms latent Gaussian noise matrices into mock transaction sequence frames.

        Args:
            noise: Input tensor of shape (batch_size, sequence_length, noise_dim)
        Returns:
            Synthetic transaction sequence tensor of shape (batch_size, sequence_length, feature_dim)
        """
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        lstm_out, _ = self.lstm(noise)

        # Project each temporal sequence step output down to the target transactional dimension
        features = self.fc(lstm_out)

        # Apply normalization activation limit bound mapping
        return self.tanh(features)

    def sample_noise(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """
        Helper method to generate correctly shaped random noise distributions.
        """
        return torch.randn(
            batch_size, self.sequence_length, self.noise_dim, device=device
        )
