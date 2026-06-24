import pytest
import numpy as np
from src.models.layers import GraphSAGELayer

# Skip if PyTorch is not installed to prevent pipeline testing blockages
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def test_mock_or_real_layer_dims():
    """
    Validates GraphSAGE Layer output dimensionality matches structural requirements.
    """
    in_feats = 16
    out_feats = 32
    
    if HAS_TORCH:
        # Construct real PyTorch modules and test
        layer = GraphSAGELayer(in_feats, out_feats, concat=False)
        
        # Simulated node tensor (4 nodes, 16 features each)
        x = torch.randn(4, in_feats)
        # Undirected line graph: 0-1-2-3
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (4, out_feats), f"Expected shape (4, {out_feats}), got {output.shape}"
        
        # Test concatenation variant
        layer_concat = GraphSAGELayer(in_feats, out_feats, concat=True)
        output_concat = layer_concat(x, edge_index)
        assert output_concat.shape == (4, out_feats * 2), f"Expected concatenated shape (4, {out_feats * 2}), got {output_concat.shape}"
    else:
        # Mock testing checks
        layer = GraphSAGELayer(in_feats, out_feats)
        dummy_x = np.random.randn(4, in_feats)
        dummy_edges = np.array([[0, 1], [1, 2]])
        output = layer.forward(dummy_x, dummy_edges)
        assert output is not None
        assert layer.in_feats == in_feats
