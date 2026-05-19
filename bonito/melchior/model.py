"""
Bonito Melchior-style CTC and CTC-CRF model.

This implements a Melchior-like architecture using Bonito framework patterns:
- Stem network for input projection
- Alternating Mamba and Transformer blocks
- CTC or CTC-CRF output head for basecalling
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
# Main Model - CTC Version
# ============================================================================

class MelchiorCTCModel(Module):
    """
    Melchior-like model with Bonito-style interface using standard CTC.
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


# ============================================================================
# CTC-CRF Components (based on bonito.crf)
# ============================================================================

try:
    from koi.ctc import SequenceDist, Max, Log, semiring
    from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
    KOI_AVAILABLE = True
except ImportError:
    KOI_AVAILABLE = False
    SequenceDist = Module  # Fallback to Module when koi not available
    
    # Define stubs for type hints when koi is not available
    class _LogSemiring:
        pass
    Log = _LogSemiring()
    Max = _LogSemiring()
    semiring = type('semiring', (), {})


@register
class MelchiorCTC_CRF(SequenceDist):
    """
    CTC-CRF distribution for Melchior model.
    Based on bonito.crf.CTC_CRF implementation.
    """
    def __init__(self, state_len, alphabet):
        if not KOI_AVAILABLE:
            raise ImportError("koi library required for CTC-CRF. Install with: pip install koi")
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        a_traceback = traceback.argmax(2)
        moves = (a_traceback % len(self.alphabet)) != 0
        paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
        return torch.where(moves, paths, 0)

    def path_to_str(self, path):
        import numpy as np
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


@register
class LinearCRFEncoderForMelchior(Module):
    """
    Linear CRF encoder adapted for Melchior architecture.
    Projects from embed_dim to CRF scores.
    """
    def __init__(self, in_features, n_base, state_len, scale=5.0, blank_score=2.0, expand_blanks=True):
        super().__init__()
        self.in_features = in_features
        self.n_base = n_base
        self.state_len = state_len
        self.scale = scale
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        
        n_score = (n_base + 1) * (n_base ** state_len) if expand_blanks else (n_base + 1) * (n_base ** state_len)
        
        self.linear = nn.Linear(in_features, n_score, bias=False)
        
        # Initialize with small values
        nn.init.normal_(self.linear.weight, 0, 0.01)
        
        # Optionally boost blank scores
        if blank_score is not None and expand_blanks:
            with torch.no_grad():
                # Set blank score bias
                blank_positions = torch.arange(0, n_score, n_base + 1)
                self.linear.weight.data[:, blank_positions] += blank_score
    
    def forward(self, x):
        # x: [seq_len, batch, in_features] or [batch, seq_len, in_features]
        # Output: [seq_len, batch, n_score]
        if x.dim() == 3 and x.size(0) > x.size(1):
            # Already [seq_len, batch, features]
            return self.linear(x)
        else:
            # [batch, seq_len, features] -> [seq_len, batch, n_score]
            return self.linear(x).transpose(0, 1)
    
    def to_dict(self, include_weights=False):
        res = {
            "in_features": self.in_features,
            "n_base": self.n_base,
            "state_len": self.state_len,
            "scale": self.scale,
            "blank_score": self.blank_score,
            "expand_blanks": self.expand_blanks,
        }
        if include_weights:
            raise NotImplementedError
        return res


# ============================================================================
# Main Model - CTC-CRF Version
# ============================================================================

class MelchiorCRFModel(Module):
    """
    Melchior-like model with CTC-CRF output.
    Uses the same backbone as CTC version but with CRF head.
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
        n_base = len(self.alphabet) - 1  # Exclude blank
        
        # CRF-specific config
        crf_config = config.get('crf', {})
        state_len = crf_config.get('state_len', 5)
        scale = crf_config.get('scale', 5.0)
        blank_score = crf_config.get('blank_score', 2.0)
        expand_blanks = crf_config.get('expand_blanks', True)
        
        self.embed_dim = embed_dim
        self.output_length = output_length
        self.stride = 1
        self.n_pre_context_bases = state_len - 1
        self.n_post_context_bases = 1
        
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
        
        # CRF head instead of simple linear head
        self.crf_encoder = LinearCRFEncoderForMelchior(
            in_features=embed_dim,
            n_base=n_base,
            state_len=state_len,
            scale=scale,
            blank_score=blank_score,
            expand_blanks=expand_blanks
        )
        
        # CTC-CRF distribution
        self.seqdist = MelchiorCTC_CRF(
            state_len=state_len,
            alphabet=self.alphabet
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
        
        Returns:
            CRF scores [output_length, batch, n_scores]
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
        
        # CRF encoder
        x = self.crf_encoder(x)
        
        return x
    
    def loss(self, scores, targets, target_lengths, **kwargs):
        """CTC-CRF loss."""
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)
    
    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        """Decode using CRF Viterbi."""
        if not KOI_AVAILABLE:
            raise ImportError("koi library required for CTC-CRF decoding")
        
        # Compute posteriors
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        
        return [self.seqdist.path_to_str(tb.cpu().numpy()) for tb in tracebacks]
    
    @property
    def features(self):
        """Return number of output features (for compatibility)."""
        return self.embed_dim


# ============================================================================
# Model Factory
# ============================================================================

def Model(config):
    """
    Factory function to create Melchior model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MelchiorCTCModel or MelchiorCRFModel based on config
    """
    model_type = config.get('model_type', 'ctc')
    
    if model_type == 'crf':
        return MelchiorCRFModel(config)
    else:
        return MelchiorCTCModel(config)
