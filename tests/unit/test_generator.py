"""
Unit tests for FraudTransactionGenerator (LSTM-based adversarial sequence generator).
Covers: output shape, tanh output range, noise sampling, variable batch sizes,
deterministic behaviour with fixed seeds, and gradient flow.
"""

import pytest
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
pytestmark = requires_torch  # All tests in this file require torch


@pytest.fixture
def generator():
    from src.models.generator_lstm import FraudTransactionGenerator

    return FraudTransactionGenerator(
        noise_dim=32,
        hidden_dim=64,
        sequence_length=5,
        feature_dim=20,
        num_layers=2,
    )


class TestFraudTransactionGeneratorShape:
    def test_forward_output_shape(self, generator):
        """Output must be (batch_size, sequence_length, feature_dim)."""
        batch_size = 8
        noise = generator.sample_noise(batch_size)
        out = generator(noise)
        assert out.shape == (
            batch_size,
            generator.sequence_length,
            generator.feature_dim,
        ), f"Expected ({batch_size}, {generator.sequence_length}, {generator.feature_dim}), got {out.shape}"

    def test_sample_noise_shape(self, generator):
        """sample_noise must return (batch_size, sequence_length, noise_dim)."""
        noise = generator.sample_noise(batch_size=4)
        assert noise.shape == (4, generator.sequence_length, generator.noise_dim)

    def test_single_sample_batch(self, generator):
        """Edge case: batch_size = 1."""
        noise = generator.sample_noise(batch_size=1)
        out = generator(noise)
        assert out.shape == (1, generator.sequence_length, generator.feature_dim)

    def test_large_batch(self, generator):
        """Batch size of 128 must not cause shape errors."""
        noise = generator.sample_noise(batch_size=128)
        out = generator(noise)
        assert out.shape == (128, generator.sequence_length, generator.feature_dim)


class TestFraudTransactionGeneratorValues:
    def test_output_bounded_by_tanh(self, generator):
        """All output values must be in (-1, 1] due to tanh activation."""
        noise = generator.sample_noise(batch_size=32)
        out = generator(noise)
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_output_not_all_zeros(self, generator):
        """Sanity check: generator should produce non-trivial outputs."""
        noise = generator.sample_noise(batch_size=8)
        out = generator(noise)
        assert out.abs().sum().item() > 0.0, "Generator must produce non-zero outputs"

    def test_different_noise_different_output(self, generator):
        """Different noise seeds must produce different transaction sequences."""
        torch.manual_seed(0)
        noise1 = generator.sample_noise(batch_size=4)
        torch.manual_seed(999)
        noise2 = generator.sample_noise(batch_size=4)
        out1 = generator(noise1)
        out2 = generator(noise2)
        assert not torch.equal(
            out1, out2
        ), "Different noise must produce different outputs"

    def test_deterministic_with_fixed_seed(self, generator):
        """Same noise tensor must produce identical output (generator is deterministic given noise)."""
        generator.eval()
        torch.manual_seed(42)
        noise = generator.sample_noise(batch_size=4)
        out1 = generator(noise)
        out2 = generator(noise)
        torch.testing.assert_close(out1, out2)


class TestFraudTransactionGeneratorGradients:
    def test_gradients_flow_through_generator(self, generator):
        """Gradients must propagate back through LSTM → fc → tanh."""
        noise = generator.sample_noise(batch_size=4)
        noise.requires_grad_(True)
        out = generator(noise)
        loss = out.mean()
        loss.backward()
        assert noise.grad is not None, "Gradient must flow back to input noise"
        assert noise.grad.abs().sum().item() > 0.0

    def test_parameters_exist(self, generator):
        """Generator must have trainable parameters."""
        params = list(generator.parameters())
        assert len(params) > 0, "Generator must have trainable parameters"
        total = sum(p.numel() for p in params)
        assert total > 0


class TestFraudTransactionGeneratorConfig:
    def test_custom_dimensions(self):
        """Generator should respect custom noise_dim, hidden_dim, feature_dim."""
        from src.models.generator_lstm import FraudTransactionGenerator

        gen = FraudTransactionGenerator(
            noise_dim=16, hidden_dim=32, sequence_length=3, feature_dim=10, num_layers=1
        )
        noise = gen.sample_noise(batch_size=2)
        out = gen(noise)
        assert out.shape == (2, 3, 10)

    def test_default_dimensions_match_project_spec(self):
        """Default feature_dim=345 matches the GNN input size (339 Vesta + 6 extras)."""
        from src.models.generator_lstm import FraudTransactionGenerator

        gen = FraudTransactionGenerator()
        assert gen.feature_dim == 345
        assert gen.noise_dim == 64
        assert gen.sequence_length == 10
