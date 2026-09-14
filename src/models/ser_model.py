import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class ConvBlock(nn.Module):
    """Convolutional Residual Block with Batch Normalization and Dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + res)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class AttentionPooling(nn.Module):
    """Attention mechanism over the time dimension of the BiLSTM output."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context

class TamilSERModel(nn.Module):
    """
    3-Channel Acoustic Residual-CNN + BiLSTM + Attention Deep Learning Model
    for Tamil Speech Emotion Recognition.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 3-Channel Input (Log-Mel, Delta, Delta-Delta)
        self.block1 = ConvBlock(in_channels=3, out_channels=32, dropout=0.1)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, dropout=0.15)
        self.block3 = ConvBlock(in_channels=64, out_channels=128, dropout=0.2)

        # Mel frequency dimension after 3 pooling layers: 64 // 8 = 8
        conv_out_mel_dim = config.n_mels // 8
        lstm_input_dim = 128 * conv_out_mel_dim  # 1024

        # Bidirectional LSTM for Temporal Emotion Dynamics
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0
        )

        bilstm_out_dim = config.lstm_hidden_size * 2
        self.layer_norm = nn.LayerNorm(bilstm_out_dim)
        self.attention = AttentionPooling(bilstm_out_dim)

        # Emotion Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(64, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, 3, n_mels, time_steps)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Reshape for LSTM: (batch_size, time_steps, channels * freq)
        batch_size, channels, freq, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * freq)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits

def build_model(config: Config) -> nn.Module:
    return TamilSERModel(config)
