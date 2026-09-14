import time
from pathlib import Path
from typing import Dict, Any, List

try:
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
except ImportError:
    torch = None
    nn = object
    GradScaler = None
    autocast = None
    AdamW = None
    CosineAnnealingWarmRestarts = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.config import Config
from src.data.vocabulary import tamil_vocab
from src.training.metrics import calculate_cer, calculate_wer

class ASRTrainer:
    """
    Training Engine for Tamil Automatic Speech Recognition (ASR) with CTCLoss.
    """
    def __init__(self, model: nn.Module, config: Config, train_loader, val_loader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # CTC Loss Criterion
        self.criterion = nn.CTCLoss(blank=config.blank_id, zero_infinity=True)

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6
        )

        # Mixed Precision Scaler
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler(device_type, enabled=config.use_amp)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_cer": [],
            "val_wer": []
        }
        self.best_val_cer = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1:02d}/{self.config.epochs} [Train]", leave=False)

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        for batch in pbar:
            specs = batch["spectrograms"].to(self.config.device)
            input_lengths = batch["input_lengths"].to(self.config.device)
            targets = batch["targets"].to(self.config.device)
            target_lengths = batch["target_lengths"].to(self.config.device)

            self.optimizer.zero_grad()

            with autocast(device_type=device_type, enabled=self.config.use_amp):
                log_probs, output_lengths = self.model(specs, input_lengths)
                # CTCLoss expects shape: (time, batch, vocab_size)
                ctc_log_probs = log_probs.permute(1, 0, 2)
                loss = self.criterion(ctc_log_probs, targets, output_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / max(len(self.train_loader), 1)

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_hypotheses = []
        all_references = []

        with torch.no_grad():
            for batch in self.val_loader:
                specs = batch["spectrograms"].to(self.config.device)
                input_lengths = batch["input_lengths"].to(self.config.device)
                targets = batch["targets"].to(self.config.device)
                target_lengths = batch["target_lengths"].to(self.config.device)
                ref_texts = batch["tamil_texts"]

                log_probs, output_lengths = self.model(specs, input_lengths)
                ctc_log_probs = log_probs.permute(1, 0, 2)
                loss = self.criterion(ctc_log_probs, targets, output_lengths, target_lengths)
                total_loss += loss.item()

                hyp_texts = tamil_vocab.ctc_decode_batch(log_probs)
                all_hypotheses.extend(hyp_texts)
                all_references.extend(ref_texts)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        cer = calculate_cer(all_hypotheses, all_references)
        wer = calculate_wer(all_hypotheses, all_references)

        return {"loss": avg_loss, "cer": cer, "wer": wer}

    def save_checkpoint(self, path: Path, epoch: int, cer: float, loss: float):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cer": cer,
            "loss": loss,
            "vocab_size": self.config.vocab_size,
            "config": self.config
        }, path)

    def fit(self):
        print(f"🚀 Starting Tamil Speech-to-Text Training ({self.config.epochs} epochs)...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            val_loss = val_metrics["loss"]
            val_cer = val_metrics["cer"]
            val_wer = val_metrics["wer"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_cer"].append(val_cer)
            self.history["val_wer"].append(val_wer)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save best checkpoint based on Character Error Rate (CER)
            if val_cer <= self.best_val_cer or epoch == 0:
                self.best_val_cer = val_cer
                best_path = self.config.checkpoint_dir / "best_tamil_asr_model.pth"
                self.save_checkpoint(best_path, epoch, val_cer, val_loss)
                print(f"--> Saved best checkpoint: Val CER = {val_cer*100:.2f}% | WER = {val_wer*100:.2f}%")

            print(
                f"Epoch [{epoch+1:02d}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val CER: {val_cer*100:.2f}% | "
                f"Val WER: {val_wer*100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        elapsed = time.time() - start_time
        print(f"\n✨ ASR Training completed in {elapsed//60:.0f}m {elapsed%60:.0f}s. Best Val CER: {self.best_val_cer*100:.2f}%")
        self.plot_curves()

    def plot_curves(self):
        """Plots training loss and validation CER/WER curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), facecolor='#0B0F19')

        # Loss Plot
        axes[0].set_facecolor('#0B0F19')
        axes[0].plot(self.history["train_loss"], label="Train CTC Loss", color="#F59E0B", lw=2)
        axes[0].plot(self.history["val_loss"], label="Val CTC Loss", color="#6366F1", lw=2, linestyle="--")
        axes[0].set_title("CTC Training & Validation Loss", color="white", fontsize=12, pad=10)
        axes[0].set_xlabel("Epoch", color="#94A3B8")
        axes[0].set_ylabel("Loss", color="#94A3B8")
        axes[0].tick_params(colors="#94A3B8")
        axes[0].grid(True, color="#334155", alpha=0.3)
        axes[0].legend(facecolor='#0B0F19', labelcolor='white')

        # Error Rates Plot (CER & WER)
        axes[1].set_facecolor('#0B0F19')
        cer_pct = [c * 100 for c in self.history["val_cer"]]
        wer_pct = [w * 100 for w in self.history["val_wer"]]
        axes[1].plot(cer_pct, label="Character Error Rate (CER %)", color="#EF4444", lw=2)
        axes[1].plot(wer_pct, label="Word Error Rate (WER %)", color="#10B981", lw=2, linestyle="--")
        axes[1].set_title("Validation Error Rates (CER / WER)", color="white", fontsize=12, pad=10)
        axes[1].set_xlabel("Epoch", color="#94A3B8")
        axes[1].set_ylabel("Error Rate (%)", color="#94A3B8")
        axes[1].tick_params(colors="#94A3B8")
        axes[1].grid(True, color="#334155", alpha=0.3)
        axes[1].legend(facecolor='#0B0F19', labelcolor='white')

        plt.tight_layout()
        plot_path = self.config.output_dir / "tamil_asr_training_curves.png"
        plt.savefig(plot_path, dpi=200, facecolor='#0B0F19')
        plt.close()
        print(f"📈 Training curves saved to: {plot_path}")
