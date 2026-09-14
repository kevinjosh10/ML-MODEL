try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    nn_Module = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    nn_Module = object

from typing import Tuple, Dict, Any

from src.config import Config
from src.data.vocabulary import tamil_vocab

class ASRConvBlock(nn_Module):
    """Convolutional Block with Residual Connection, BatchNorm, and Time-Frequency Sub-sampling."""
    def __init__(self, in_channels: int, out_channels: int, pool_time: bool = True, dropout: float = 0.1):
        super().__init__()
        if nn is not None:
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
                
            pool_stride = (2, 2) if pool_time else (2, 1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=pool_stride)
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + res)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class TamilASRModel(nn_Module):
    """
    Deep Residual-CNN + Multi-layer Bidirectional LSTM + CTC Projection Network
    for Tamil Automatic Speech Recognition (Audio-to-Text).
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # 1. 2D Convolutional Acoustic Sub-sampling Front-end
        # Input shape: (batch_size, 1, n_mels=80, time_steps)
        self.conv1 = ASRConvBlock(in_channels=1, out_channels=32, pool_time=True, dropout=0.1)   # Time / 2, Freq / 2 (40)
        self.conv2 = ASRConvBlock(in_channels=32, out_channels=64, pool_time=True, dropout=0.15) # Time / 4, Freq / 4 (20)
        self.conv3 = ASRConvBlock(in_channels=64, out_channels=128, pool_time=False, dropout=0.2) # Time / 4, Freq / 8 (10)

        # Mel frequency dimension after 3 pooling layers: 80 // 8 = 10
        conv_out_freq_dim = config.n_mels // 8
        rnn_input_dim = 128 * conv_out_freq_dim  # 128 * 10 = 1280

        self.input_projection = nn.Linear(rnn_input_dim, config.lstm_hidden_size)

        # 2. Deep Multi-layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=config.lstm_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0
        )

        bilstm_dim = config.lstm_hidden_size * 2
        self.layer_norm = nn.LayerNorm(bilstm_dim)
        
        # 3. Dense Classification Head to Tamil Character Vocabulary
        self.fc = nn.Sequential(
            nn.Linear(bilstm_dim, config.lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size, self.vocab_size)
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Spectrogram tensor (batch_size, 1, n_mels, time_steps)
            input_lengths: Acoustic sequence lengths (batch_size,)
        Returns:
            log_probs: CTC log probabilities (batch_size, time_steps_subsampled, vocab_size)
            output_lengths: Sub-sampled sequence lengths (batch_size,)
        """
        # 1. Conv Feature Extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Reshape: (batch_size, channels, freq, time) -> (batch_size, time, channels * freq)
        batch_size, channels, freq, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * freq)

        # 2. RNN Projection & BiLSTM
        x = self.input_projection(x)
        x, _ = self.bilstm(x)
        x = self.layer_norm(x)

        # 3. Vocabulary CTC Projection
        logits = self.fc(x)  # (batch_size, time_steps, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate sub-sampled output lengths (divided by 4 due to 2 time pooling layers)
        if input_lengths is not None:
            output_lengths = torch.clamp(input_lengths // 4, min=1)
        else:
            output_lengths = torch.tensor([time_steps] * batch_size, device=x.device, dtype=torch.long)

        return log_probs, output_lengths

def build_asr_model(config: Config) -> TamilASRModel:
    """Builds and initializes the Tamil ASR Neural Network."""
    model = TamilASRModel(config)
    return model.to(config.device)
