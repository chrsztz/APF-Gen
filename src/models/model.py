from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.crf import CRF


class LocalMultiHeadAttention(nn.Module):
    """Windowed multi-head self-attention.

    Each position attends only to positions within ±window of itself.
    Uses relative position bias so the model can learn distance-aware patterns.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, window: int = 10, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window = window

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable relative position bias: 2*window+1 possible relative distances
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * window + 1))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) bool
        Returns:
            out: (B, T, D) — residual + attention output
            attn_weights: (B, T) — mean attention weight per position (for visualization)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product: (B, H, T, T)
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Build local window mask: position i can attend to j if |i-j| <= window
        pos = torch.arange(T, device=x.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
        local_mask = dist <= self.window  # (T, T)

        # Add relative position bias
        rel_idx = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.window, self.window) + self.window  # (T, T) in [0, 2w]
        rel_bias = self.rel_pos_bias[:, rel_idx]  # (H, T, T)
        attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch

        # Apply local window mask
        attn = attn.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply padding mask (only mask keys, not queries)
        pad_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        attn = attn.masked_fill(~pad_mask, -1e9)

        attn_weights = torch.softmax(attn, dim=-1)  # (B, H, T, T)
        # Padded query positions → all-masked rows → NaN after softmax; zero them out
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection
        out = residual + out

        # Return mean attention weight per position for visualization
        avg_attn = attn_weights.mean(dim=1).mean(dim=-1)  # (B, T)
        return out, avg_attn


class FingeringModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phys_dim: int,
        hidden_size: int = 128,
        cnn_channels: int = 64,
        cnn_layers: int = 2,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 11,
        use_attention: bool = True,
        attn_heads: int = 4,
        attn_window: int = 10,
        use_crf: bool = False,
        use_ar: bool = False,
        ar_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ar = use_ar

        convs = []
        in_ch = input_dim
        for _ in range(cnn_layers):
            convs.append(nn.Conv1d(in_ch, cnn_channels, kernel_size=7, padding=3))
            convs.append(nn.BatchNorm1d(cnn_channels))
            convs.append(nn.LeakyReLU())
            convs.append(nn.Dropout(dropout))
            in_ch = cnn_channels
        self.conv = nn.Sequential(*convs)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2
        self.use_attention = use_attention
        self.local_attn = (
            LocalMultiHeadAttention(
                embed_dim=lstm_out_dim,
                num_heads=attn_heads,
                window=attn_window,
                dropout=dropout,
            )
            if use_attention
            else None
        )

        # AR: embed previous finger and concatenate with BiLSTM+attn output
        # Index num_classes = <start> token
        if use_ar:
            self.finger_emb = nn.Embedding(num_classes + 1, ar_embed_dim)
            head_in_dim = lstm_out_dim + ar_embed_dim
        else:
            self.finger_emb = None
            head_in_dim = lstm_out_dim

        self.main_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.phys_head = nn.Sequential(
            nn.Linear(phys_dim, phys_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phys_dim * 2, num_classes),
        )
        self.use_crf = use_crf
        self.crf = CRF(num_classes) if use_crf else None

    def _encode(self, main_feats: torch.Tensor, mask: torch.Tensor):
        """Shared CNN → BiLSTM → LocalAttn encoder."""
        x = main_feats.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        fused, _ = self.lstm(x)
        if self.use_attention and self.local_attn is not None:
            fused, attn_w = self.local_attn(fused, mask)
        else:
            attn_w = torch.zeros_like(mask, dtype=fused.dtype)
        return fused, attn_w

    def forward(
        self,
        main_feats: torch.Tensor,
        phys_feats: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        labels: (batch, seq) ground-truth finger indices — only used during training
                when use_ar=True for teacher forcing. Pass None for inference.
        """
        fused, attn_w = self._encode(main_feats, mask)

        if self.use_ar:
            if labels is not None:
                # Teacher forcing: shift labels right, prepend <start>
                B, T = fused.shape[:2]
                start = torch.full((B, 1), self.num_classes, device=fused.device, dtype=torch.long)
                prev = torch.cat([start, labels[:, :-1].clamp(min=0)], dim=1)  # (B, T)
                prev_emb = self.finger_emb(prev)  # (B, T, ar_embed_dim)
                main_logits = self.main_head(torch.cat([fused, prev_emb], dim=-1))
            else:
                # Greedy AR inference: step-by-step
                B, T = fused.shape[:2]
                prev_tok = torch.full((B,), self.num_classes, device=fused.device, dtype=torch.long)
                out_list = []
                for t in range(T):
                    prev_emb = self.finger_emb(prev_tok)  # (B, ar_embed_dim)
                    step_in = torch.cat([fused[:, t], prev_emb], dim=-1)
                    logit_t = self.main_head(step_in)  # (B, num_classes)
                    out_list.append(logit_t.unsqueeze(1))
                    prev_tok = logit_t.argmax(dim=-1)
                main_logits = torch.cat(out_list, dim=1)
        else:
            main_logits = self.main_head(fused)

        phys_logits = self.phys_head(phys_feats)
        return main_logits, phys_logits, attn_w
