"""
Bonito Melchior-style CTC model.

This implements a Melchior-like architecture using Bonito framework patterns:
- Stem network for input projection
- Alternating Mamba and Transformer blocks
- CTC output head for basecalling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from bonito.nn import register, to_dict, from_dict, layers


# ============================================================================
# Model Components
# ============================================================================

@register
class MelchiorStem(Module):
    """
    Stem network similar to Melchior's FastEmbed.
    Projects input signal to higher dimensional space.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 512, embed_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-4),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim, eps=1e-4),
            nn.GELU(approximate='tanh')
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        x = self.conv_stack(x)
        # Transpose to [batch, seq_len, embed_dim] for transformer/mamba blocks
        return x.transpose(1, 2)
    
    def to_dict(self, include_weights=False):
        res = {
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class MambaBlock(Module):
    """
    Simplified Mamba-style state space model block.
    Based on Melchior's implementation.
    
    Note: This is a simplified version. For production use, consider using
    the official mamba_ssm library for the selective scan operation.
    """
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.d_model = dim
        self.d_state = d_state
        self.d_inner = expand * dim
        self.drop_path_rate = drop_path
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Initialize dt parameters
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
        # Drop path
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        batch, seq_len, dim = x.shape
        
        # Project and split for gate mechanism
        x_proj = self.in_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        
        # Simplified SSM (without actual selective scan for simplicity)
        # In production, you would use mamba_ssm library
        x_ssm = self._simplified_ssm(x)
        
        # Gate mechanism
        x_ssm = x_ssm * F.sigmoid(gate)
        
        # Output projection
        x_out = self.out_proj(x_ssm)
        
        return residual + self.drop_path(x_out)
    
    def _simplified_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified SSM computation - placeholder for actual mamba_ssm"""
        # This is a placeholder - in real implementation use mamba_ssm
        # For now, use a simple projection
        x_proj = self.x_proj(x)
        return x
    
    def to_dict(self, include_weights=False):
        res = {
            "dim": self.d_model,
            "d_state": self.d_state,
            "expand": self.d_inner // self.d_model,
            "drop_path": self.drop_path_rate,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class MelchiorTransformerBlock(Module):
    """
    Standard Transformer block with self-attention and MLP.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, 
                 drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop
        self.drop_path_rate = drop_path
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=drop,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate='tanh'),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(
            self.norm1(x), 
            self.norm1(x), 
            self.norm1(x),
            need_weights=False
        )
        x = x + self.drop_path(attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        
        return x
    
    def to_dict(self, include_weights=False):
        res = {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "drop": self.drop_rate,
            "drop_path": self.drop_path_rate,
        }
        if include_weights:
            raise NotImplementedError
        return res


@register
class MelchiorHead(Module):
    """
    Output head that projects to class scores.
    Similar to Melchior's Head module.
    """
    def __init__(self, in_features: int, seq_len: int, out_seq_len: int, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.num_classes = num_classes
        
        # Sequence compression/expansion
        self.seq_proj = nn.Linear(seq_len, out_seq_len)
        
        # Feature projection to classes
        self.out_proj = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features]
        x = x.permute(0, 2, 1)  # [batch, in_features, seq_len]
        x = self.seq_proj(x)    # [batch, in_features, out_seq_len]
        x = x.permute(0, 2, 1)  # [batch, out_seq_len, in_features]
        
        # Project to class scores
        x = self.out_proj(x)    # [batch, out_seq_len, num_classes]
        
        # Permute to [out_seq_len, batch, num_classes] for CTC
        x = x.permute(1, 0, 2)
        
        return x
    
    def to_dict(self, include_weights=False):
        res = {
            "in_features": self.in_features,
            "seq_len": self.seq_len,
            "out_seq_len": self.out_seq_len,
            "num_classes": self.num_classes,
        }
        if include_weights:
            raise NotImplementedError
        return res


# ============================================================================
# Main Model
# ============================================================================

class Model(Module):
    """
    Melchior-like model with Bonito-style interface.
    Alternates between Mamba and Transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        model_config = config.get('model_config', {})
        in_channels = model_config.get('in_channels', 1)
        embed_dim = model_config.get('embed_dim', 768)
        depth = model_config.get('depth', 20)
        num_heads = model_config.get('num_heads', 8)
        mlp_ratio = model_config.get('mlp_ratio', 4.0)
        drop_rate = model_config.get('drop_rate', 0.0)
        drop_path_rate = model_config.get('drop_path_rate', 0.1)
        output_length = model_config.get('output_length', 420)
        input_seq_len = model_config.get('input_seq_len', 4096)
        
        # Get alphabet from config
        self.alphabet = config.get('labels', {}).get('labels', ['N', 'A', 'C', 'G', 'T'])
        num_classes = len(self.alphabet)
        
        self.embed_dim = embed_dim
        self.output_length = output_length
        self.stride = 1  # For compatibility with bonito data loading
        
        # Stem
        self.stem = MelchiorStem(
            in_channels=in_channels,
            hidden_dim=embed_dim // 2,
            embed_dim=embed_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, input_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Alternating Mamba and Transformer blocks
        self.blocks = nn.ModuleList([
            MambaBlock(dim=embed_dim, drop_path=dpr[i]) if i % 2 == 0
            else MelchiorTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head
        self.head = MelchiorHead(
            in_features=embed_dim,
            seq_len=input_seq_len,
            out_seq_len=output_length,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
        
        Returns:
            Log probabilities [output_length, batch, num_classes]
        """
        # Stem embedding
        x = self.stem(x)  # [batch, seq_len, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Output head
        x = self.head(x)
        
        # Log softmax for CTC
        x = F.log_softmax(x, dim=-1)
        
        return x
    
    def loss(self, log_probs, targets, lengths):
        """CTC loss with label smoothing."""
        return self.ctc_label_smoothing_loss(log_probs, targets, lengths)
    
    def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        """
        CTC loss with label smoothing.
        """
        T, N, C = log_probs.shape
        weights = weights or torch.cat([
            torch.tensor([0.4]),
            (0.1 / (C - 1)) * torch.ones(C - 1)
        ])
        log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64, device=log_probs.device)
        loss = F.ctc_loss(
            log_probs.to(torch.float32),
            targets,
            log_probs_lengths,
            lengths,
            blank=0,
            reduction='mean',
            zero_infinity=True
        )
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'total_loss': loss + label_smoothing_loss, 'loss': loss, 'label_smooth_loss': label_smoothing_loss}
    
    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        """
        Decode CTC output to sequences.
        
        Uses greedy decoding by default. For beam search, install fast_ctc_decode.
        """
        try:
            from fast_ctc_decode import beam_search, viterbi_search
            import numpy as np
            
            x_np = x.exp().cpu().numpy().astype(np.float32)
            if beamsize == 1 or qscores:
                seq, path = viterbi_search(x_np, self.alphabet, qscores, 1.0, 0.0)
            else:
                seq, path = beam_search(x_np, self.alphabet, beamsize, threshold)
            if return_path:
                return seq, path
            return seq
        except ImportError:
            # Fallback to greedy decoding
            predictions = x.argmax(dim=-1).transpose(0, 1)  # [batch, output_length]
            
            decoded_sequences = []
            for pred in predictions:
                sequence = []
                prev_token = -1
                for token in pred:
                    token_idx = token.item()
                    if token_idx != prev_token and token_idx != 0:  # 0 is blank
                        sequence.append(self.alphabet[token_idx])
                    prev_token = token_idx
                decoded_sequences.append(''.join(sequence))
            
            return decoded_sequences if len(decoded_sequences) > 1 else decoded_sequences[0]
    
    @property
    def features(self):
        """Return number of output features (for compatibility)."""
        return self.embed_dim
