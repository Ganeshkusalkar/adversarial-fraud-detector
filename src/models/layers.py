import logging

logger = logging.getLogger(__name__)

# Fallback for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch is not installed. Defining mock Layer classes.")

    # Define placeholder mock base class
    class nn:
        class Module:
            def __init__(self):
                pass


if HAS_TORCH:

    class GraphSAGELayer(nn.Module):
        """
        Inductive GraphSAGE Message-Passing Layer.
        Performs node representation aggregation over localized neighborhoods.
        """

        def __init__(self, in_feats: int, out_feats: int, concat: bool = True):
            super(GraphSAGELayer, self).__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.concat = concat

            # Linear transform for self representations
            self.linear_self = nn.Linear(in_feats, out_feats, bias=False)
            # Linear transform for neighborhood aggregation
            self.linear_neigh = nn.Linear(in_feats, out_feats, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_feats))

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """
            x: Node feature tensor [num_nodes, in_feats]
            edge_index: Graph connection indexes [2, num_edges]
            """
            num_nodes = x.size(0)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # Aggregate neighborhood node states (mean aggregation)
            # 1. Initialize neighborhood representations with zeros
            neigh_feats = torch.zeros((num_nodes, self.in_feats), device=x.device)
            # 2. Count degrees for averaging
            degrees = torch.zeros(num_nodes, device=x.device).index_add_(
                0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float)
            )
            degrees = torch.clamp(degrees, min=1.0)

            # 3. Sum up source features onto destination nodes
            neigh_feats = neigh_feats.index_add(0, dst_idx, x[src_idx])
            neigh_feats = neigh_feats / degrees.unsqueeze(1)

            # Linear transformation
            out_self = self.linear_self(x)
            out_neigh = self.linear_neigh(neigh_feats)

            if self.concat:
                out = torch.cat([out_self, out_neigh], dim=-1)
                # If we concatenate, the dimension is 2 * out_feats
            else:
                out = out_self + out_neigh + self.bias

            return F.relu(out)

else:

    class GraphSAGELayer(nn.Module):
        """Mock implementation of GraphSAGELayer for development environment."""

        def __init__(self, in_feats: int, out_feats: int, concat: bool = True):
            super().__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.concat = concat
            logger.info("Initializing mock GraphSAGELayer class...")

        def forward(self, x, edge_index):
            return x
