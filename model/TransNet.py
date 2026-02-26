"""
TransNet.py  (v2 — SEED-IV adapted)
=====================================
Identical architecture to the original EEG-TransNet paper.

Fixes / notes vs the version in the old adaptation:
  1. x.squeeze(dim=2) — explicit dim argument prevents silent collapse
     of the batch dimension when batch_size == 1 (e.g. last test batch).
  2. Default changed to embed_dim=64 and num_channels=62 to match SEED-IV.
     (BCI-2a users should keep embed_dim=32, num_channels=22.)
  3. VarPool computes variance in float32 before log to prevent
     overflow/NaN under AMP (float16).

Everything else — temporal conv bank, spatial conv, transformer stack,
conv encoder, FC — is unmodified from the original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim ** 0.5
    attn   = F.softmax(scores, dim=-1)
    out    = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn


class VarPoold(nn.Module):
    """Variance pooling with log transform.  AMP-safe."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride

    def forward(self, x):
        T         = x.shape[2]
        out_shape = (T - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            s = i * self.stride
            slc = x[:, :, s:s + self.kernel_size]
            # Compute in float32 to avoid FP16 overflow under AMP
            var_val = slc.float().var(dim=-1, keepdim=True)
            log_var = torch.log(torch.clamp(var_val, 1e-6, 1e6)).to(x.dtype)
            out.append(log_var)
        return torch.cat(out, dim=-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k   = d_model // n_head
        self.d_v   = d_model // n_head
        self.n_head = n_head
        self.w_q   = nn.Linear(d_model, n_head * self.d_k)
        self.w_k   = nn.Linear(d_model, n_head * self.d_k)
        self.w_v   = nn.Linear(d_model, n_head * self.d_v)
        self.w_o   = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(self.w_k(key),   'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(self.w_v(value), 'b n (h d) -> b h n d', h=self.n_head)
        out, _ = attention(q, k, v)
        out    = rearrange(out, 'b h q d -> b q (h d)')
        return self.dropout(self.w_o(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1     = nn.Linear(d_model, d_hidden)
        self.act     = nn.GELU()
        self.w_2     = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.act(self.w_1(x)))))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio,
                 attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.mha        = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.ff         = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        res = self.layernorm1(x)
        x   = x + self.mha(res, res, res)
        res = self.layernorm2(x)
        return x + self.ff(res)


class TransNet(nn.Module):
    """
    Attention-based CNN with multi-modal temporal fusion.
    Ma et al., Computers in Biology and Medicine, 2024.

    Default args match SEED-IV configuration (embed_dim=64, num_channels=62).
    For BCI-2a use embed_dim=32, num_channels=22, num_samples=1000.
    """
    def __init__(self,
                 num_classes:  int   = 4,
                 num_samples:  int   = 800,
                 num_channels: int   = 62,
                 embed_dim:    int   = 64,    # SEED-IV default (was 32)
                 pool_size:    int   = 50,
                 pool_stride:  int   = 15,
                 num_heads:    int   = 8,
                 fc_ratio:     int   = 4,
                 depth:        int   = 4,
                 attn_drop:    float = 0.5,
                 fc_drop:      float = 0.5):
        super().__init__()

        # ── Temporal conv bank (4 scales) ──────────────────────────────
        self.temp_conv1 = nn.Conv2d(1, embed_dim // 4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, embed_dim // 4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, embed_dim // 4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, embed_dim // 4, (1, 65), padding=(0, 32))
        self.bn1        = nn.BatchNorm2d(embed_dim)

        # ── Spatial conv (cross-channel mixing) ────────────────────────
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))
        self.bn2          = nn.BatchNorm2d(embed_dim)
        self.elu          = nn.ELU()

        # ── Temporal pooling (two modalities) ──────────────────────────
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        T_embed = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout()

        # ── Shared transformer encoders ────────────────────────────────
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio,
                               attn_drop, fc_drop)
            for _ in range(depth)
        ])

        # ── Convolutional encoder (fuses avg/var modalities) ───────────
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(T_embed, T_embed, (2, 1)),
            nn.BatchNorm2d(T_embed),
            nn.ELU()
        )

        # ── Classifier ─────────────────────────────────────────────────
        self.classify = nn.Linear(embed_dim * T_embed, num_classes)

    def forward(self, x):
        # x: (B, 62, 800)
        x = x.unsqueeze(dim=1)    # (B, 1, 62, 800)

        # Multi-scale temporal features
        x = self.bn1(torch.cat([
            self.temp_conv1(x),
            self.temp_conv2(x),
            self.temp_conv3(x),
            self.temp_conv4(x),
        ], dim=1))                 # (B, embed_dim, 62, 800)

        # Spatial mixing
        x = self.elu(self.bn2(self.spatial_conv(x)))  # (B, embed_dim, 1, 800)
        x = x.squeeze(dim=2)      # (B, embed_dim, 800)  ← explicit dim!

        # Two-modality pooling
        x1 = self.dropout(self.avg_pool(x))  # (B, embed_dim, T_embed)
        x2 = self.dropout(self.var_pool(x))  # (B, embed_dim, T_embed)

        # Reshape for transformer: (B, T_embed, embed_dim)
        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        # Shared transformer stack
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)

        # Convolutional encoder fuses the two modalities
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x  = self.conv_encoder(torch.cat([x1, x2], dim=2))  # (B, T_embed, 1, embed_dim)

        return self.classify(x.reshape(x.size(0), -1))
