import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.config import Config
from src.utils.visualizer import plot_training_history

class Trainer:
    """
    Trainer for Tamil Speech Emotion Recognition model.
    Optimized for Google Colab GPUs (T4/V100/A100) with AMP & Cosine Annealing.
    """
    def __init__(self, model: nn.Module, config: Config, train_loader, val_loader=None):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(5, config.epochs // 3),
            T_mult=1,
            eta_min=1e-5
        )
        
        self.scaler = GradScaler(enabled=config.use_amp)
        self.best_val_acc = -1.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1:02d}/{self.config.epochs:02d} [Train]", leave=False)
        for specs, labels in pbar:
            specs = specs.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast(enabled=True):
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                self.optimizer.step()
            
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1:02d}/{self.config.epochs:02d} [Val]", leave=False)
        for specs, labels in pbar:
            specs = specs.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)
            
            if self.config.use_amp:
                with autocast(enabled=True):
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def fit(self):
        print("=" * 65)
        print(f"🚀 Training Tamil SER on: {self.config.device.upper()} (AMP: {self.config.use_amp})")
        print("=" * 65)
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            t_loss, t_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(t_loss)
            self.history['train_acc'].append(t_acc)
            
            val_str = ""
            if self.val_loader:
                v_loss, v_acc = self.validate_epoch(epoch)
                self.history['val_loss'].append(v_loss)
                self.history['val_acc'].append(v_acc)
                val_str = f" | Val Loss: {v_loss:.4f} - Val Acc: {v_acc:.2f}%"
                
                if v_acc >= self.best_val_acc:
                    self.best_val_acc = v_acc
                    patience_counter = 0
                    self.save_checkpoint("best_tamil_ser_model.pth", epoch, v_acc)
                    print(f"--> Saved best checkpoint: Val Acc = {v_acc:.2f}%")
                else:
                    patience_counter += 1
            else:
                self.save_checkpoint("best_tamil_ser_model.pth", epoch, t_acc)

            self.save_checkpoint("latest_tamil_ser_model.pth", epoch, t_acc)
            self.scheduler.step()

            lr_curr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:02d}/{self.config.epochs:02d}] "
                  f"Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.2f}%"
                  f"{val_str} | LR: {lr_curr:.6f}")
                  
            if patience_counter >= self.config.early_stopping_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

        elapsed = time.time() - start_time
        print(f"\n✨ Training completed in {elapsed//60:.0f}m {elapsed%60:.0f}s. Best Val Acc: {max(self.best_val_acc, 0.0):.2f}%")
        
        # Guarantee checkpoint exists
        best_ckpt = self.config.checkpoint_dir / "best_tamil_ser_model.pth"
        if not best_ckpt.exists():
            self.save_checkpoint("best_tamil_ser_model.pth", self.config.epochs, self.best_val_acc)

        curves_path = self.config.output_dir / "tamil_ser_training_curves.png"
        plot_training_history(self.history, save_path=curves_path)
        print(f"📈 Training curves saved to: {curves_path}")

    def save_checkpoint(self, filename: str, epoch: int, accuracy: float):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.config.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'classes': self.config.classes
        }, filepath)
