import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class AttentionPooling(nn.Module):
    """Attention mechanism over the time dimension of the LSTM output."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        weights = self.attn(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_dim)
        return context

class TamilSERModel(nn.Module):
    """
    Hybrid CNN + BiLSTM + Attention Deep Learning architecture for
    Speech Emotion Recognition from Log-Mel Spectrograms.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 2D CNN Acoustic Feature Extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.15)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2)
        )

        # Mel frequency dimension after 3 pooling layers: 64 / 8 = 8
        conv_out_mel_dim = config.n_mels // 8
        lstm_input_dim = 128 * conv_out_mel_dim  # 128 * 8 = 1024

        # Bidirectional LSTM for Temporal Emotion Dynamics
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0
        )

        # Attention Pooling over temporal frames
        bilstm_out_dim = config.lstm_hidden_size * 2
        self.attention = AttentionPooling(bilstm_out_dim)

        # Emotion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, 1, n_mels, time_steps)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x shape: (batch_size, 128, n_mels//8, time_steps//8)

        # Reshape for LSTM: (batch_size, time_steps//8, 128 * n_mels//8)
        batch_size, channels, freq, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * freq)

        lstm_out, _ = self.lstm(x)  # (batch_size, time_steps, lstm_hidden_size * 2)
        context = self.attention(lstm_out)  # (batch_size, lstm_hidden_size * 2)
        logits = self.classifier(context)   # (batch_size, num_classes)
        return logits

def build_model(config: Config) -> nn.Module:
    return TamilSERModel(config)
