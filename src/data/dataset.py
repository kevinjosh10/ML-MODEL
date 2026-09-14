from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json

try:
    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    np = None
    Dataset = object
    DataLoader = object

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    def train_test_split(arr, test_size=0.2, random_state=42):
        n = int(len(arr) * (1 - test_size))
        return arr[:n], arr[n:]

from src.config import Config
from src.data.vocabulary import tamil_vocab
from src.data.audio_preprocessing import AudioPreprocessor
from src.data.tamil_corpus import generate_tamil_asr_dataset, TAMIL_ASR_CORPUS

class TamilASRDataset(Dataset):
    """
    Paired Tamil Speech-to-Text Dataset.
    Loads audio Log-Mel Spectrogram and encoded Tamil character tokens.
    """
    def __init__(self, records: List[Dict[str, Any]], config: Config, is_train: bool = True):
        self.records = records
        self.config = config
        self.is_train = is_train
        self.preprocessor = AudioPreprocessor(config, is_train=is_train)
        self.vocab = tamil_vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        file_path = Path(item["audio_path"])
        tamil_text = item["tamil_text"]

        # 1. Load and extract Log-Mel Spectrogram features
        # Shape: (3, n_mels, time_steps) or (n_mels, time_steps)
        features = self.preprocessor.process_file(file_path)
        # Use first channel if multi-channel: (n_mels, time_steps)
        if features.dim() == 3:
            spec = features[0] # (n_mels, time_steps)
        else:
            spec = features
            
        time_steps = spec.shape[-1]

        # 2. Encode text transcript to integer character IDs
        encoded_targets = self.vocab.encode(tamil_text)
        targets_tensor = torch.tensor(encoded_targets, dtype=torch.long)
        target_len = len(encoded_targets)

        return {
            "spectrogram": spec,               # (n_mels, time_steps)
            "input_len": time_steps,           # Original acoustic frame count
            "targets": targets_tensor,         # (target_len,)
            "target_len": target_len,          # Number of characters
            "tamil_text": tamil_text,
            "id": item.get("id", f"sample_{idx}")
        }

def asr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that dynamically pads variable-length spectrograms and text targets.
    Prepares input_lengths and target_lengths for PyTorch CTCLoss.
    """
    n_mels = batch[0]["spectrogram"].shape[0]
    max_input_len = max(item["input_len"] for item in batch)
    max_target_len = max(item["target_len"] for item in batch)

    batch_size = len(batch)
    
    # Padded spectrogram tensor: (batch_size, 1, n_mels, max_input_len)
    padded_specs = torch.zeros(batch_size, 1, n_mels, max_input_len, dtype=torch.float32)
    # Padded targets tensor: (batch_size, max_target_len)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)

    input_lengths = []
    target_lengths = []
    tamil_texts = []
    sample_ids = []

    for i, item in enumerate(batch):
        spec = item["spectrogram"]
        t_len = item["input_len"]
        padded_specs[i, 0, :, :t_len] = spec
        input_lengths.append(t_len)

        target = item["targets"]
        tgt_len = item["target_len"]
        padded_targets[i, :tgt_len] = target
        target_lengths.append(tgt_len)

        tamil_texts.append(item["tamil_text"])
        sample_ids.append(item["id"])

    return {
        "spectrograms": padded_specs,                                      # (B, 1, n_mels, T_max)
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),   # (B,)
        "targets": padded_targets,                                         # (B, L_max)
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long), # (B,)
        "tamil_texts": tamil_texts,
        "ids": sample_ids
    }

def get_asr_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, List[Dict[str, Any]]]:
    """
    Loads or generates the paired Tamil ASR speech dataset and splits into Train/Val/Test loaders.
    """
    manifest_path = config.data_dir / "manifest.json"
    
    if not manifest_path.exists():
        manifest_path = generate_tamil_asr_dataset(config.data_dir, config.sample_rate, max_samples=35)

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    def resolve_valid(rec_list):
        valid = []
        for r in rec_list:
            raw_p = Path(r["audio_path"])
            if raw_p.exists():
                r_copy = dict(r)
                r_copy["audio_path"] = str(raw_p)
                valid.append(r_copy)
            else:
                cand = config.data_dir / "wavs" / raw_p.name
                if cand.exists():
                    r_copy = dict(r)
                    r_copy["audio_path"] = str(cand)
                    valid.append(r_copy)
                else:
                    cand2 = config.data_dir / raw_p.name
                    if cand2.exists():
                        r_copy = dict(r)
                        r_copy["audio_path"] = str(cand2)
                        valid.append(r_copy)
        return valid

    valid_records = resolve_valid(records)
    
    if len(valid_records) < 4:
        # Regenerate if corrupt or missing
        generate_tamil_asr_dataset(config.data_dir, config.sample_rate, max_samples=35)
        with open(manifest_path, "r", encoding="utf-8") as f:
            new_records = json.load(f)
        valid_records = resolve_valid(new_records)

    # Train / Val / Test Split
    train_records, test_records = train_test_split(valid_records, test_size=0.2, random_state=config.seed)
    train_records, val_records = train_test_split(train_records, test_size=0.15, random_state=config.seed)

    train_dataset = TamilASRDataset(train_records, config, is_train=True)
    val_dataset = TamilASRDataset(val_records, config, is_train=False)
    test_dataset = TamilASRDataset(test_records, config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=asr_collate_fn,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=asr_collate_fn,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=asr_collate_fn,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, valid_records
