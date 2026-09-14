try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    import torchaudio.functional as AF
    import torch.nn.functional as F
except ImportError:
    torch = None
    torchaudio = None
    T = None
    AF = None
    F = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import librosa
except ImportError:
    librosa = None

from pathlib import Path
from typing import Union, Optional
from src.config import Config

class AudioPreprocessor:
    """
    State-of-the-Art Audio Acoustic Preprocessor:
    Extracts Log-Mel Spectrogram features for Tamil Speech-to-Text ASR.
    """
    def __init__(self, config: Config, is_train: bool = False):
        self.config = config
        self.sample_rate = config.sample_rate
        self.is_train = is_train
        
        # Mel filterbank
        if T is not None:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                power=2.0
            )
            self.amplitude_to_db = T.AmplitudeToDB()
            self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
            self.time_mask = T.TimeMasking(time_mask_param=12)
        else:
            self.mel_transform = None
            self.amplitude_to_db = None
            self.freq_mask = None
            self.time_mask = None

    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Loads audio file, converts to mono, resamples to 16kHz, and normalizes length."""
        file_path = str(file_path)
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            y, sr = librosa.load(file_path, sr=None, mono=False)
            if y.ndim == 1:
                waveform = torch.from_numpy(y).unsqueeze(0)
            else:
                waveform = torch.from_numpy(y)

        # Convert stereo/multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to target sample rate
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pre-emphasis filter to boost speech formants
        waveform = self._apply_preemphasis(waveform)

        # Fix length
        waveform = self._fix_length(waveform)
        return waveform

    def _apply_preemphasis(self, waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
        """Boosts higher frequencies (consonants and vocal formants)."""
        if waveform.shape[-1] < 2:
            return waveform
        return torch.cat([waveform[:, :1], waveform[:, 1:] - coeff * waveform[:, :-1]], dim=-1)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads or crops waveform to exactly target_samples."""
        num_samples = waveform.shape[1]
        if num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif num_samples < self.target_samples:
            pad_amount = self.target_samples - num_samples
            waveform = F.pad(waveform, (0, pad_amount))
        return waveform

    def extract_mel_spectrogram(self, waveform: torch.Tensor, augment: Optional[bool] = None) -> torch.Tensor:
        """
        Extracts 3-Channel Feature Tensor:
        Channel 0: Normalized Log-Mel Spectrogram
        Channel 1: Delta (Velocity of frequency transitions)
        Channel 2: Delta-Delta (Acceleration of emotional energy)
        Output shape: (3, n_mels, time_steps)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)  # (1, n_mels, time_steps)
        
        apply_aug = self.is_train if augment is None else augment
        if apply_aug:
            log_mel_spec = self.freq_mask(log_mel_spec)
            log_mel_spec = self.time_mask(log_mel_spec)

        # Compute Deltas (Velocity & Acceleration)
        delta1 = AF.compute_deltas(log_mel_spec)
        delta2 = AF.compute_deltas(delta1)

        # Stack into 3 Channels: (3, n_mels, time_steps)
        features = torch.cat([log_mel_spec, delta1, delta2], dim=0)

        # Per-channel mean & variance normalization
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        normalized_features = (features - mean) / std
        return normalized_features

    def process_file(self, file_path: Union[str, Path], augment: Optional[bool] = None) -> torch.Tensor:
        waveform = self.load_audio(file_path)
        return self.extract_mel_spectrogram(waveform, augment=augment)
