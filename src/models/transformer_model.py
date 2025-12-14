import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerFingering(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phys_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 11,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.main_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.phys_head = nn.Sequential(
            nn.Linear(phys_dim, phys_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phys_dim * 2, num_classes),
        )

    def forward(self, main_feats: torch.Tensor, phys_feats: torch.Tensor, mask: torch.Tensor):
        x = self.input_proj(main_feats)
        x = self.pos_enc(x)
        # transformer expects key_padding_mask where True = pad
        key_padding_mask = ~mask
        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)
        main_logits = self.main_head(enc)
        phys_logits = self.phys_head(phys_feats)
        attn_dummy = torch.zeros_like(mask, dtype=x.dtype)
        return main_logits, phys_logits, attn_dummy


