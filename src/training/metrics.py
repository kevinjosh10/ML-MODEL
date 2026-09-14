from typing import List, Dict, Any, Tuple
try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

from src.config import Config
from src.data.vocabulary import tamil_vocab

def levenshtein_distance(ref: List[Any], hyp: List[Any]) -> int:
    """Computes minimum edit distance (Levenshtein) between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Deletion
                                   dp[i][j - 1],    # Insertion
                                   dp[i - 1][j - 1])# Substitution

    return int(dp[n][m])

def calculate_cer(hypotheses: List[str], references: List[str]) -> float:
    """Computes Character Error Rate (CER)."""
    total_dist = 0
    total_chars = 0
    
    for hyp, ref in zip(hypotheses, references):
        total_dist += levenshtein_distance(list(ref), list(hyp))
        total_chars += max(len(ref), 1)

    return float(total_dist / total_chars) if total_chars > 0 else 0.0

def calculate_wer(hypotheses: List[str], references: List[str]) -> float:
    """Computes Word Error Rate (WER)."""
    total_dist = 0
    total_words = 0

    for hyp, ref in zip(hypotheses, references):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_dist += levenshtein_distance(ref_words, hyp_words)
        total_words += max(len(ref_words), 1)

    return float(total_dist / total_words) if total_words > 0 else 0.0

def evaluate_asr_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, config: Config) -> Dict[str, Any]:
    """
    Evaluates the Tamil ASR model on validation/test data using CER and WER.
    """
    model.eval()
    all_hypotheses = []
    all_references = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = torch.nn.CTCLoss(blank=config.blank_id, zero_infinity=True)

    with torch.no_grad():
        for batch in data_loader:
            specs = batch["spectrograms"].to(config.device)
            input_lengths = batch["input_lengths"].to(config.device)
            targets = batch["targets"].to(config.device)
            target_lengths = batch["target_lengths"].to(config.device)
            ref_texts = batch["tamil_texts"]

            log_probs, output_lengths = model(specs, input_lengths)
            
            # log_probs for CTCLoss needs shape: (time, batch, vocab_size)
            ctc_log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(ctc_log_probs, targets, output_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            hyp_texts = tamil_vocab.ctc_decode_batch(log_probs)
            all_hypotheses.extend(hyp_texts)
            all_references.extend(ref_texts)

    avg_loss = total_loss / max(num_batches, 1)
    cer = calculate_cer(all_hypotheses, all_references)
    wer = calculate_wer(all_hypotheses, all_references)
    accuracy = max(0.0, (1.0 - cer) * 100.0)

    return {
        "loss": avg_loss,
        "cer": cer,
        "wer": wer,
        "character_accuracy": accuracy,
        "sample_hypotheses": all_hypotheses[:5],
        "sample_references": all_references[:5]
    }
