import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Union, Optional
from src.config import Config

class AudioPreprocessor:
    """
    Handles audio loading, resampling, fixed-length padding/truncation,
    Log-Mel Spectrogram extraction, and SpecAugment data augmentation.
    """
    def __init__(self, config: Config, is_train: bool = False):
        self.config = config
        self.sample_rate = config.sample_rate
        self.target_samples = config.target_samples
        self.is_train = is_train
        
        # PyTorch Mel Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # SpecAugment for training augmentation
        self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
        self.time_mask = T.TimeMasking(time_mask_param=12)

    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Loads an audio file, converts to mono, resamples to target sample rate,
        and pads/crops to the target duration.
        """
        file_path = str(file_path)
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
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

    def extract_mel_spectrogram(self, waveform: torch.Tensor, augment: Optional[bool] = None) -> torch.Tensor:
        """
        Converts raw waveform into a normalized Log-Mel Spectrogram tensor.
        Output shape: (1, n_mels, time_steps)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Apply SpecAugment if training
        apply_aug = self.is_train if augment is None else augment
        if apply_aug:
            log_mel_spec = self.freq_mask(log_mel_spec)
            log_mel_spec = self.time_mask(log_mel_spec)

        # Normalization
        mean = log_mel_spec.mean()
        std = log_mel_spec.std() + 1e-6
        normalized_spec = (log_mel_spec - mean) / std
        return normalized_spec

    def process_file(self, file_path: Union[str, Path], augment: Optional[bool] = None) -> torch.Tensor:
        waveform = self.load_audio(file_path)
        return self.extract_mel_spectrogram(waveform, augment=augment)
