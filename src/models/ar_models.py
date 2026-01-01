import torch
import torch.nn as nn
import torch.nn.functional as F


class ArLSTM(nn.Module):
    """
    Simple autoregressive LSTM decoder:
    - condition on input features (main)
    - previous predicted finger embedding is fed at each step
    - during training/inference we use greedy prev prediction (no teacher forcing)
    """

    def __init__(self, input_dim: int, phys_dim: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 11, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.main_proj = nn.Linear(input_dim, hidden_size)
        self.prev_emb = nn.Embedding(num_classes, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + phys_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, main_feats: torch.Tensor, phys_feats: torch.Tensor, mask: torch.Tensor):
        b, t, _ = main_feats.shape
        x_main = self.main_proj(main_feats)
        # init prev token as zeros (class 0 embedding)
        prev_tok = torch.zeros((b,), device=main_feats.device, dtype=torch.long)
        outputs = []
        h, c = None, None
        for step in range(t):
            prev_emb = self.prev_emb(prev_tok)  # (b, hidden)
            step_in = torch.cat([x_main[:, step, :], prev_emb], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(step_in, (h, c)) if h is not None else self.lstm(step_in)
            logits = self.head(torch.cat([out.squeeze(1), phys_feats[:, step, :]], dim=-1))
            outputs.append(logits.unsqueeze(1))
            prev_tok = logits.argmax(dim=-1)
        logits_all = torch.cat(outputs, dim=1)
        # dummy attn
        attn = torch.zeros_like(mask, dtype=logits_all.dtype)
        return logits_all, phys_feats.new_zeros(logits_all.shape), attn


class ArGNN(nn.Module):
    """
    Placeholder simple AR GNN-like model:
    Use 1D conv + BiLSTM but name distinct for experiment tagging.
    """

    def __init__(self, input_dim: int, phys_dim: int, hidden_size: int = 128, num_classes: int = 11, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2 + phys_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, main_feats: torch.Tensor, phys_feats: torch.Tensor, mask: torch.Tensor):
        x = main_feats.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        logits = self.head(torch.cat([lstm_out, phys_feats], dim=-1))
        attn = torch.zeros_like(mask, dtype=logits.dtype)
        return logits, phys_feats.new_zeros(logits.shape), attn


