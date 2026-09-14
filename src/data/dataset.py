import os
import math
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.config import Config
from src.data.audio_preprocessing import AudioPreprocessor

class TamilSpeechEmotionDataset(Dataset):
    """
    PyTorch Dataset for Tamil Speech Emotion audio files.
    Expects directory structure:
        data_dir/
            happy/
                sample1.wav, sample2.wav, ...
            sad/
                sample1.wav, ...
            angry/
                ...
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

def create_sample_dataset(data_dir: Path, sample_rate: int = 16000, duration: float = 3.0, samples_per_class: int = 20):
    """
    Generates synthetic sample audio waveforms for each emotion class with distinct
    acoustic profiles (pitch, modulation, tempo) to test and run the Colab pipeline immediately.
    """
    classes = ["happy", "sad", "angry", "neutral", "fear", "surprised"]
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Acoustic profile configurations for speech emotions
    profiles = {
        "happy": {"base_f": 260, "mod_f": 5, "mod_d": 40, "noise": 0.05, "envelope": "rising"},
        "sad": {"base_f": 140, "mod_f": 1.5, "mod_d": 10, "noise": 0.02, "envelope": "decaying"},
        "angry": {"base_f": 320, "mod_f": 8, "mod_d": 60, "noise": 0.15, "envelope": "harsh"},
        "neutral": {"base_f": 180, "mod_f": 2, "mod_d": 15, "noise": 0.03, "envelope": "flat"},
        "fear": {"base_f": 290, "mod_f": 12, "mod_d": 50, "noise": 0.08, "envelope": "tremolo"},
        "surprised": {"base_f": 340, "mod_f": 4, "mod_d": 80, "noise": 0.04, "envelope": "peaked"}
    }

    for emotion in classes:
        emotion_dir = data_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        prof = profiles[emotion]
        for i in range(samples_per_class):
            file_name = emotion_dir / f"tamil_sample_{emotion}_{i+1:03d}.wav"
            if file_name.exists():
                continue
                
            # Synthesize synthetic vocal formant simulation
            jitter = np.random.uniform(0.9, 1.1)
            f0 = prof["base_f"] * jitter
            freq_mod = f0 + prof["mod_d"] * np.sin(2 * np.pi * prof["mod_f"] * t)
            phase = 2 * np.pi * np.cumsum(freq_mod) / sample_rate
            
            # Harmonic tones simulating speech formants
            signal = (
                0.6 * np.sin(phase) +
                0.3 * np.sin(2 * phase) +
                0.15 * np.sin(3 * phase)
            )
            
            # Add subtle acoustic noise
            signal += np.random.normal(0, prof["noise"], num_samples)
            
            # Apply amplitude envelope
            if prof["envelope"] == "decaying":
                env = np.exp(-t * 0.8)
            elif prof["envelope"] == "harsh":
                env = np.clip(np.sin(np.pi * t / duration) * 1.5, 0, 1)
            elif prof["envelope"] == "tremolo":
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 6 * t)
            elif prof["envelope"] == "peaked":
                env = np.exp(-((t - 1.5) ** 2) / 0.5)
            else:
                env = np.sin(np.pi * t / duration)
                
            signal = signal * env
            signal = signal / (np.max(np.abs(signal)) + 1e-6)
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
    """Loads dataset and splits into train (80%), validation (10%), test (10%)."""
    preprocessor = AudioPreprocessor(config)
    
    file_paths, labels = scan_dataset(config.data_dir, config.classes)
    
    # If dataset is empty, create sample starter dataset
    if len(file_paths) == 0:
        print("No existing audio files found in data directory. Generating starter sample dataset...")
        create_sample_dataset(config.data_dir, config.sample_rate, config.duration, samples_per_class=25)
        file_paths, labels = scan_dataset(config.data_dir, config.classes)
        print(f"Generated {len(file_paths)} sample audio clips across {len(config.classes)} emotions.")

    dataset = TamilSpeechEmotionDataset(file_paths, labels, preprocessor)
    
    total_size = len(dataset)
    val_size = max(1, int(total_size * 0.15))
    test_size = max(1, int(total_size * 0.10))
    train_size = total_size - val_size - test_size
    
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
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
