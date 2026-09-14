from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import torch

@dataclass
class Config:
    # Experiment & Paths
    experiment_name: str = "tamil_ser_cnn_bilstm"
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    
    # Audio signal processing parameters
    sample_rate: int = 16000           # 16 kHz standard
    duration: float = 3.0              # 3 seconds fixed duration
    target_samples: int = 16000 * 3    # 48000 samples
    n_mels: int = 64                   # Mel filterbanks
    n_fft: int = 1024                  # FFT window
    hop_length: int = 512              # Hop size
    
    # Emotion Classes & Labels
    emotion_map: Dict[str, str] = field(default_factory=lambda: {
        "happy": "மகிழ்ச்சி (Happy)",
        "sad": "சோகம் (Sad)",
        "angry": "கோபம் (Angry)",
        "neutral": "இயல்பு (Neutral)",
        "fear": "பயம் (Fear)",
        "surprised": "ஆச்சரியம் (Surprised)"
    })
    
    # Clean ASCII / Phonetic labels for Matplotlib charts
    plot_labels: Dict[str, str] = field(default_factory=lambda: {
        "happy": "Happy (Magizhchi)",
        "sad": "Sad (Sogam)",
        "angry": "Angry (Kobam)",
        "neutral": "Neutral (Iyalbu)",
        "fear": "Fear (Bayam)",
        "surprised": "Surprised (Aachariyam)"
    })
    
    classes: List[str] = field(default_factory=lambda: [
        "happy", "sad", "angry", "neutral", "fear", "surprised"
    ])
    
    # Model architecture
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    
    # Training Hyperparameters
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    
    # Hardware & Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True if torch.cuda.is_available() else False
    seed: int = 42

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
