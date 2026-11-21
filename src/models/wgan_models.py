import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequence modeling.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class ConditionalGenerator(nn.Module):
    """
    BiLSTM Generator for Conditional WGAN-GP.
    Generates synthetic transaction temporal sequences.
    """
    def __init__(self, noise_dim=100, cond_dim=10, seq_len=10, feature_dim=30, hidden_dim=128, num_layers=2):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Projection layer for conditioning
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.LeakyReLU(0.2),
        )
        
        # BiLSTM processing input noise + projected conditioning
        # Input size: noise_dim + 32
        self.lstm = nn.LSTM(
            input_size=noise_dim + 32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output mapping back to transaction feature space
        # BiLSTM output dim is hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, z, cond):
        # z shape: [batch_size, seq_len, noise_dim]
        # cond shape: [batch_size, cond_dim]
        
        batch_size = z.size(0)
        c_proj = self.cond_proj(cond) # [batch_size, 32]
        c_expanded = c_proj.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch_size, seq_len, 32]
        
        # Concatenate noise and conditioning along features dimension
        lstm_in = torch.cat([z, c_expanded], dim=-1) # [batch_size, seq_len, noise_dim + 32]
        
        lstm_out, _ = self.lstm(lstm_in) # [batch_size, seq_len, hidden_dim * 2]
        out = self.fc(lstm_out) # [batch_size, seq_len, feature_dim]
        return out

class TemporalTransformerDiscriminator(nn.Module):
    """
    Temporal Transformer Discriminator (Critic) for Conditional WGAN-GP.
    Evaluates transaction sequences to determine realism.
    """
    def __init__(self, feature_dim=30, cond_dim=10, seq_len=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.seq_len = seq_len
        
        # Projection layer for conditioning
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.LeakyReLU(0.2)
        )
        
        # Projection layer for input features + conditioning
        self.input_proj = nn.Linear(feature_dim + 32, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to single critic score (no activation for WGAN critic)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, cond):
        # x shape: [batch_size, seq_len, feature_dim]
        # cond shape: [batch_size, cond_dim]
        
        batch_size = x.size(0)
        c_proj = self.cond_proj(cond) # [batch_size, 32]
        c_expanded = c_proj.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch_size, seq_len, 32]
        
        # Combine input features with conditioning
        combined = torch.cat([x, c_expanded], dim=-1) # [batch_size, seq_len, feature_dim + 32]
        
        feat_proj = self.input_proj(combined) # [batch_size, seq_len, d_model]
        feat_encoded = self.pos_encoder(feat_proj) # [batch_size, seq_len, d_model]
        
        trans_out = self.transformer_encoder(feat_encoded) # [batch_size, seq_len, d_model]
        
        # Flatten representation
        flat = trans_out.contiguous().view(batch_size, -1) # [batch_size, d_model * seq_len]
        
        score = self.critic_head(flat) # [batch_size, 1]
        return score

def compute_gradient_penalty(critic, real_samples, fake_samples, cond, device="cpu"):
    """
    Computes Gradient Penalty (GP) for Conditional WGAN-GP.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Random interpolation between real and fake
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Get scores for interpolates
    d_outputs = critic(interpolates, cond)
    
    # Compute gradients w.r.t. interpolates
    grad_outputs = torch.ones(d_outputs.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_outputs,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients to compute L2 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # GP is distance from 1 squared
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
