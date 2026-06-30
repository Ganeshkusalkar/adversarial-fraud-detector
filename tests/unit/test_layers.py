"""
Expanded unit tests for GNN model components:
  - GraphSAGELayer (custom implementation in src/models/layers.py)
  - FraudGNN discriminator (src/models/discriminator_gnn.py)
  - FraudTransactionGenerator LSTM (src/models/generator_lstm.py)
"""
import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.layers import GraphSAGELayer

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ===========================================================================
# GraphSAGELayer Tests
# ===========================================================================

class TestGraphSAGELayer:

    @requires_torch
    def test_output_shape_no_concat(self):
        """Non-concatenation mode: output dim == out_feats."""
        layer = GraphSAGELayer(in_feats=16, out_feats=32, concat=False)
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"

    @requires_torch
    def test_output_shape_concat(self):
        """Concatenation mode: output dim == 2 * out_feats."""
        layer = GraphSAGELayer(in_feats=16, out_feats=32, concat=True)
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    @requires_torch
    def test_output_nonnegative_relu(self):
        """ReLU activation means all output values should be >= 0."""
        layer = GraphSAGELayer(in_feats=8, out_feats=16, concat=False)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, edge_index)
        assert (out >= 0).all(), "ReLU output must be non-negative"

    @requires_torch
    def test_isolated_nodes_no_edges(self):
        """Zero edges: neighbourhood aggregation falls back to self-transform only."""
        layer = GraphSAGELayer(in_feats=8, out_feats=16, concat=False)
        x = torch.randn(3, 8)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (3, 16), "Must handle isolated-node graphs"

    @requires_torch
    def test_single_node_self_loop(self):
        """Single node with a self-loop shouldn't crash."""
        layer = GraphSAGELayer(in_feats=4, out_feats=8, concat=False)
        x = torch.randn(1, 4)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (1, 8)

    def test_mock_layer_attributes_without_torch(self):
        """Even without PyTorch, layer stores in_feats / out_feats."""
        layer = GraphSAGELayer(in_feats=4, out_feats=8)
        assert layer.in_feats == 4
        assert layer.out_feats == 8


# ===========================================================================
# FraudGNN Discriminator Tests
# ===========================================================================

@requires_torch
class TestFraudGNN:

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.models.discriminator_gnn import FraudGNN
        self.FraudGNN = FraudGNN
        self.in_channels = 16
        self.hidden = 32

    def _make_graph(self, n_nodes=6, n_edges=8):
        x = torch.randn(n_nodes, self.in_channels)
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        return x, edge_index

    def test_logits_shape(self):
        """forward() returns logits of shape [num_nodes, 2]."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        model.eval()
        x, edge_index = self._make_graph()
        logits = model(x, edge_index)
        assert logits.shape == (x.shape[0], 2), f"Expected ({x.shape[0]}, 2), got {logits.shape}"

    def test_fraud_proba_range(self):
        """predict_fraud_proba() must return values in [0, 1]."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        x, edge_index = self._make_graph()
        proba = model.predict_fraud_proba(x, edge_index)
        assert proba.shape == (x.shape[0],)
        assert (proba >= 0.0).all() and (proba <= 1.0).all(), "Probabilities must be in [0, 1]"

    def test_classify_returns_booleans(self):
        """classify() must return a boolean tensor."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        x, edge_index = self._make_graph()
        predictions = model.classify(x, edge_index)
        assert predictions.dtype == torch.bool, "classify() must return bool tensor"
        assert predictions.shape == (x.shape[0],)

    def test_custom_threshold_respected(self):
        """Setting threshold=0.0 should flag all nodes as fraud."""
        model = self.FraudGNN(
            in_channels=self.in_channels, hidden_channels=self.hidden, decision_threshold=0.0
        )
        x, edge_index = self._make_graph()
        predictions = model.classify(x, edge_index)
        assert predictions.all(), "With threshold=0.0 all nodes should be flagged"

    def test_threshold_1_0_no_fraud(self):
        """Setting threshold=1.0 should flag nothing as fraud (prob is never exactly 1.0)."""
        model = self.FraudGNN(
            in_channels=self.in_channels, hidden_channels=self.hidden, decision_threshold=1.0
        )
        x, edge_index = self._make_graph()
        predictions = model.classify(x, edge_index)
        assert not predictions.any(), "With threshold=1.0 no node should be flagged"

    def test_no_grad_in_eval_mode(self):
        """predict_fraud_proba is decorated @torch.no_grad() — gradients must not accumulate."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        x, edge_index = self._make_graph()
        proba = model.predict_fraud_proba(x, edge_index)
        assert not proba.requires_grad, "Inference outputs must not require grad"

    def test_isolated_nodes_no_crash(self):
        """GNN must handle a graph with zero edges without crashing."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        model.eval()
        x = torch.randn(3, self.in_channels)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        proba = model.predict_fraud_proba(x, edge_index)
        assert proba.shape == (3,)

    def test_batch_mode_pooling(self):
        """With a batch vector, global_mean_pool reduces output to [num_graphs, 2]."""
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden)
        model.eval()
        # 2 graphs: 3 nodes each
        x = torch.randn(6, self.in_channels)
        edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        logits = model(x, edge_index, batch=batch)
        assert logits.shape == (2, 2), f"Expected (2, 2) with batched pooling, got {logits.shape}"

    def test_dropout_active_in_train_mode(self):
        """
        In training mode with dropout, two identical forward passes should differ.
        (This can rarely be equal by chance, but is overwhelmingly reliable in practice.)
        """
        model = self.FraudGNN(in_channels=self.in_channels, hidden_channels=self.hidden, dropout=0.5)
        model.train()
        torch.manual_seed(0)
        x = torch.randn(10, self.in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
        assert not torch.equal(out1, out2), "Dropout should make training-mode outputs stochastic"
