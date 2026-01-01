from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (batch, seq, hidden)
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # (batch, seq)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (batch, hidden)
        return context, weights


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
    ):
        super().__init__()
        convs = []
        in_ch = input_dim
        for _ in range(cnn_layers):
            convs.append(nn.Conv1d(in_ch, cnn_channels, kernel_size=3, padding=1))
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
        self.use_attention = use_attention
        self.attn = Attention(hidden_size * 2) if use_attention else None
        attn_out_dim = hidden_size * 4 if use_attention else hidden_size * 2
        self.main_head = nn.Sequential(
            nn.Linear(attn_out_dim, hidden_size),
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

    def forward(self, main_feats: torch.Tensor, phys_feats: torch.Tensor, mask: torch.Tensor):
        # main_feats: (batch, seq, input_dim)
        x = main_feats.transpose(1, 2)  # (batch, input_dim, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq, cnn_channels)
        packed_out, _ = self.lstm(x)
        if self.use_attention:
            context, attn_w = self.attn(packed_out, mask)
            context_expanded = context.unsqueeze(1).expand(-1, packed_out.size(1), -1)
            fused = torch.cat([packed_out, context_expanded], dim=-1)
        else:
            fused = packed_out
            attn_w = torch.zeros_like(mask, dtype=packed_out.dtype)
        main_logits = self.main_head(fused)
        phys_logits = self.phys_head(phys_feats)
        return main_logits, phys_logits, attn_w




