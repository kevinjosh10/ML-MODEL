from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
try:
    import torch
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    amp_default = True if torch.cuda.is_available() else False
except ImportError:
    torch = None
    device_default = "cpu"
    amp_default = False

from src.data.vocabulary import tamil_vocab

@dataclass
class Config:
    # Experiment & Paths
    experiment_name: str = "tamil_asr_cnn_bilstm_ctc"
    data_dir: Path = Path("data/asr")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    
    # Audio signal processing parameters
    sample_rate: int = 16000           # 16 kHz standard
    duration: float = 4.0              # Max duration per segment (seconds)
    n_mels: int = 80                   # Mel filterbanks for ASR
    n_fft: int = 1024                  # FFT window size
    hop_length: int = 256              # Hop size (16ms frame step)
    
    # Model architecture parameters
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    lstm_hidden_size: int = 256
    lstm_layers: int = 3
    dropout: float = 0.2
    
    # Vocabulary & CTC Parameters
    vocab_size: int = tamil_vocab.size
    blank_id: int = tamil_vocab.blank_id
    
    # Training Hyperparameters
    epochs: int = 35
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 10
    
    # Hardware & Precision
    device: str = device_default
    use_amp: bool = amp_default
    seed: int = 42

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
