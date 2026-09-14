import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Union
from src.config import Config

class AudioPreprocessor:
    """
    Handles audio loading, resampling, fixed-length padding/truncation,
    and Log-Mel Spectrogram feature extraction.
    """
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.target_samples = config.target_samples
        
        # PyTorch Mel Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Loads an audio file, converts to mono, resamples to target sample rate,
        and pads/crops to the target duration.
        """
        file_path = str(file_path)
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            # Fallback to librosa/soundfile if backend issue
            y, sr = librosa.load(file_path, sr=None, mono=False)
            if y.ndim == 1:
                waveform = torch.from_numpy(y).unsqueeze(0)
            else:
                waveform = torch.from_numpy(y)

        # Convert multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if sample rate differs
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Fix length (pad or crop)
        waveform = self._fix_length(waveform)
        return waveform

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads with zeros or crops to exactly target_samples."""
        num_samples = waveform.shape[1]
        if num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif num_samples < self.target_samples:
            pad_amount = self.target_samples - num_samples
            waveform = F.pad(waveform, (0, pad_amount))
        return waveform

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Converts raw waveform into a Log-Mel Spectrogram tensor.
        Output shape: (1, n_mels, time_steps)
        """
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Normalize to zero mean and unit variance
        mean = log_mel_spec.mean()
        std = log_mel_spec.std() + 1e-6
        normalized_spec = (log_mel_spec - mean) / std
        return normalized_spec

    def process_file(self, file_path: Union[str, Path]) -> torch.Tensor:
        waveform = self.load_audio(file_path)
        return self.extract_mel_spectrogram(waveform)
