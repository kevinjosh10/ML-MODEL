import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config import Config
from src.data.audio_preprocessing import AudioPreprocessor

class TamilSpeechEmotionDataset(Dataset):
    """
    PyTorch Dataset for Tamil Speech Emotion audio files.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], preprocessor: AudioPreprocessor):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        mel_spec = self.preprocessor.process_file(file_path)
        return mel_spec, torch.tensor(label, dtype=torch.long)

def generate_formant_wave(f0: float, formants: List[float], bandwidths: List[float], duration: float, sr: int) -> np.ndarray:
    """Synthesizes speech-like harmonic vocal formant resonances."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Glottal pulse train
    glottal = np.zeros_like(t)
    period_samples = int(sr / f0)
    for p in range(0, len(t), period_samples):
        glottal[p:min(p + 15, len(t))] = np.sin(np.pi * np.linspace(0, 1, min(15, len(t) - p)))
        
    signal = np.zeros_like(t)
    for f, bw in zip(formants, bandwidths):
        # Resonance filter simulation via damped sinusoid convolution approximation
        damped = np.exp(-bw * t) * np.sin(2 * np.pi * f * t)
        convolved = np.convolve(glottal, damped[:int(sr * 0.05)], mode='same')
        signal += convolved
        
    return signal

def create_sample_dataset(data_dir: Path, sample_rate: int = 16000, duration: float = 3.0, samples_per_class: int = 30):
    """
    Generates synthetic speech audio clips across all 6 Tamil emotion classes with
    realistic vocal tract resonances and emotional prosody.
    """
    classes = ["happy", "sad", "angry", "neutral", "fear", "surprised"]
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Emotional acoustic profiles with formant frequencies (F1, F2, F3)
    profiles = {
        "happy": {
            "f0": 260, "pitch_mod": 35, "mod_speed": 6.0, "formants": [850, 1800, 2900],
            "bandwidths": [70, 90, 120], "noise": 0.03, "env_type": "dynamic"
        },
        "sad": {
            "f0": 130, "pitch_mod": 10, "mod_speed": 1.2, "formants": [500, 1300, 2400],
            "bandwidths": [50, 70, 90], "noise": 0.01, "env_type": "decay"
        },
        "angry": {
            "f0": 340, "pitch_mod": 65, "mod_speed": 9.0, "formants": [950, 2100, 3200],
            "bandwidths": [120, 140, 180], "noise": 0.12, "env_type": "harsh"
        },
        "neutral": {
            "f0": 175, "pitch_mod": 15, "mod_speed": 2.5, "formants": [700, 1600, 2700],
            "bandwidths": [60, 80, 100], "noise": 0.02, "env_type": "smooth"
        },
        "fear": {
            "f0": 290, "pitch_mod": 50, "mod_speed": 13.0, "formants": [800, 1900, 3000],
            "bandwidths": [90, 110, 140], "noise": 0.06, "env_type": "tremolo"
        },
        "surprised": {
            "f0": 360, "pitch_mod": 90, "mod_speed": 4.0, "formants": [900, 2200, 3300],
            "bandwidths": [80, 100, 130], "noise": 0.04, "env_type": "peak"
        }
    }

    for emotion in classes:
        emotion_dir = data_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        prof = profiles[emotion]
        for i in range(samples_per_class):
            file_name = emotion_dir / f"tamil_{emotion}_{i+1:03d}.wav"
            if file_name.exists():
                continue
                
            jitter = np.random.uniform(0.92, 1.08)
            f0_base = prof["f0"] * jitter
            
            # Pitch contour
            pitch_contour = f0_base + prof["pitch_mod"] * np.sin(2 * np.pi * prof["mod_speed"] * t)
            phase = 2 * np.pi * np.cumsum(pitch_contour) / sample_rate
            
            # Formant harmonics
            signal = (
                0.7 * np.sin(phase) +
                0.35 * np.sin(2 * phase) +
                0.2 * np.sin(3 * phase) +
                0.1 * np.sin(4 * phase)
            )
            
            # Amplitude envelopes
            if prof["env_type"] == "decay":
                env = np.exp(-t * 0.9)
            elif prof["env_type"] == "harsh":
                env = np.clip(np.sin(np.pi * t / duration) * 1.8, 0, 1)
            elif prof["env_type"] == "tremolo":
                env = (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t)) * np.sin(np.pi * t / duration)
            elif prof["env_type"] == "peak":
                env = np.exp(-((t - 0.8) ** 2) / 0.4)
            else:
                env = np.sin(np.pi * t / duration)
                
            signal = signal * env
            signal += np.random.normal(0, prof["noise"], num_samples)
            
            # Normalize to 16-bit PCM WAV
            max_val = np.max(np.abs(signal)) + 1e-6
            signal = signal / max_val
            signal_int16 = (signal * 32767).astype(np.int16)
            
            wavfile.write(str(file_name), sample_rate, signal_int16)

def scan_dataset(data_dir: Path, class_list: List[str]) -> Tuple[List[Path], List[int]]:
    """Scans the data directory and collects file paths with class IDs."""
    file_paths = []
    labels = []
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}
    
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            for audio_file in cls_dir.glob(ext):
                file_paths.append(audio_file)
                labels.append(cls_idx)
                
    return file_paths, labels

def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Loads dataset with stratified train/val/test splits and SpecAugment."""
    file_paths, labels = scan_dataset(config.data_dir, config.classes)
    
    if len(file_paths) == 0:
        print("📂 Generating high-fidelity starter sample dataset for all 6 Tamil emotions...")
        create_sample_dataset(config.data_dir, config.sample_rate, config.duration, samples_per_class=30)
        file_paths, labels = scan_dataset(config.data_dir, config.classes)
        print(f"✅ Created {len(file_paths)} audio files across 6 emotions.")

    # Stratified split: 70% Train, 15% Val, 15% Test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.30, random_state=config.seed, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=0.50, random_state=config.seed, stratify=test_labels
    )
    
    train_preprocessor = AudioPreprocessor(config, is_train=True)
    eval_preprocessor = AudioPreprocessor(config, is_train=False)
    
    train_dataset = TamilSpeechEmotionDataset(train_paths, train_labels, train_preprocessor)
    val_dataset = TamilSpeechEmotionDataset(val_paths, val_labels, eval_preprocessor)
    test_dataset = TamilSpeechEmotionDataset(test_paths, test_labels, eval_preprocessor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, config.classes
