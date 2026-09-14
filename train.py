import argparse
import random
import os
import numpy as np
import torch
from src.config import Config
from src.data.dataset import get_data_loaders
from src.models import build_model
from src.training.trainer import Trainer
from src.training.metrics import evaluate_model

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tamil Speech Emotion Recognition Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Audio data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )
    set_seed(config.seed)
    
    print("=" * 60)
    print("🎭 Tamil Speech Emotion Recognition (SER) Training Pipeline")
    print(f"Device: {config.device.upper()} | AMP: {config.use_amp}")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, classes = get_data_loaders(config)
    print(f"Target Emotions: {classes}")
    
    model = build_model(config)
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.fit()
    
    # Evaluate best model on test set
    best_ckpt = config.checkpoint_dir / "best_tamil_ser_model.pth"
    if best_ckpt.exists():
        print("\nEvaluating best model on test dataset...")
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        eval_results = evaluate_model(model, test_loader, config)
        print(f"🏁 Test Accuracy: {eval_results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
