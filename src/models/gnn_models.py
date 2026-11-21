import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

def build_hetero_graph(df: pd.DataFrame) -> HeteroData:
    """
    Constructs a heterogeneous graph from a transaction DataFrame.
    The graph has a single node type 'transaction' and multiple edge types:
    - ('transaction', 'shared_card', 'transaction')
    - ('transaction', 'shared_address', 'transaction')
    - ('transaction', 'shared_email', 'transaction')
    - ('transaction', 'shared_device', 'transaction')
    - ('transaction', 'shared_ip', 'transaction')
    - ('transaction', 'shared_merchant', 'transaction')
    """
    data = HeteroData()
    
    num_nodes = len(df)
    # Node features (exclude labels and identifiers, pad/convert to float32)
    feature_cols = [c for c in df.columns if c not in ['isFraud', 'TransactionID', 'card_id', 'addr_id', 'email_id', 'device_id', 'ip_id', 'merchant_id']]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    data['transaction'].x = x
    
    if 'isFraud' in df.columns:
        data['transaction'].y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    
    # Helper to find edges based on shared attributes
    def get_edges_for_col(col_name):
        if col_name not in df.columns:
            # Return empty edge list
            return torch.empty((2, 0), dtype=torch.long)
        # Find transactions sharing the same attribute
        grouped = df.groupby(col_name).indices
        src_nodes = []
        dst_nodes = []
        for val, indices in grouped.items():
            if len(indices) > 1:
                # Create edges between all pairs in the group
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        src_nodes.extend([indices[i], indices[j]])
                        dst_nodes.extend([indices[j], indices[i]])
        if len(src_nodes) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    # Build edges for each type
    data['transaction', 'shared_card', 'transaction'].edge_index = get_edges_for_col('card_id')
    data['transaction', 'shared_address', 'transaction'].edge_index = get_edges_for_col('addr_id')
    data['transaction', 'shared_email', 'transaction'].edge_index = get_edges_for_col('email_id')
    data['transaction', 'shared_device', 'transaction'].edge_index = get_edges_for_col('device_id')
    data['transaction', 'shared_ip', 'transaction'].edge_index = get_edges_for_col('ip_id')
    data['transaction', 'shared_merchant', 'transaction'].edge_index = get_edges_for_col('merchant_id')
    
    return data

class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE Model for multi-relational transaction classification.
    """
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Layer 1
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv(in_channels, hidden_channels)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
            
        # Output Layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv(hidden_channels, out_channels)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
        return x_dict['transaction']

class HeteroGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network (GAT) for multi-relational transaction classification.
    """
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Layer 1
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
        # Hidden/Output Layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        x_dict = self.convs[0](x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
        
        # Layer 2
        x_dict = self.convs[1](x_dict, edge_index_dict)
        return x_dict['transaction']

class HeteroGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer Model utilizing TransformerConv for multi-relational graphs.
    """
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Layer 1
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = TransformerConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
        # Layer 2
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.convs.append(HeteroConv(conv_dict, aggregator='mean'))
        
    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        x_dict = self.convs[0](x_dict, edge_index_dict)
        x_dict = {key: F.gelu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
        
        # Layer 2
        x_dict = self.convs[1](x_dict, edge_index_dict)
        return x_dict['transaction']
