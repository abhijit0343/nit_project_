import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention over a sequence of LSTM outputs."""

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden_size)
        attn_scores  = self.attention(torch.tanh(lstm_outputs))   # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)              # (batch, seq_len, 1)
        context      = torch.sum(attn_weights * lstm_outputs, dim=1)  # (batch, hidden_size)
        return context, attn_weights


class ActionLSTM(nn.Module):
    """
    Bidirectional LSTM + Self-Attention model for skeleton-based action recognition.

    Args:
        input_size  : feature dimension per time-step (231 for 33 landmarks × 7 features)
        hidden_size : LSTM hidden units per direction (default 128)
        num_layers  : stacked LSTM layers (default 2)
        num_classes : number of action classes
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=10):
        super(ActionLSTM, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers

        # Layer normalisation on the raw input helps training stability
        self.input_norm = nn.LayerNorm(input_size)

        # Bidirectional LSTM — output dim is hidden_size * 2
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.4, bidirectional=True)

        # Attention works on the concatenated bidirectional output
        self.attention = SelfAttention(hidden_size * 2)

        self.dropout = nn.Dropout(0.4)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_norm(x)

        # LSTM forward pass (h0/c0 default to zeros when not provided)
        out, _ = self.lstm(x)   # out: (batch, seq_len, hidden_size * 2)

        # Attention pooling
        context, _ = self.attention(out)   # (batch, hidden_size * 2)

        out = self.dropout(context)
        out = self.fc(out)                 # (batch, num_classes)
        return out
