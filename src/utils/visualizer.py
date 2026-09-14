import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix

def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    mel_spec: np.ndarray,
    title: str = "Tamil Audio Sample",
    save_path: Optional[Path] = None
):
    """Plots waveform time-domain and Log-Mel Spectrogram frequency-domain side by side."""
    waveform = np.asarray(waveform).squeeze()
    if waveform.ndim > 1:
        waveform = waveform[0]
        
    time_axis = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))
    
    plt.figure(figsize=(14, 5))
    
    # Waveform plot
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, waveform, color='#1f77b4', alpha=0.85, linewidth=1.2)
    plt.title(f"{title} - Waveform", fontsize=12, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Mel-Spectrogram plot
    plt.subplot(1, 2, 2)
    mel_spec = np.asarray(mel_spec).squeeze()
    plt.imshow(mel_spec, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} - Log-Mel Spectrogram", fontsize=12, fontweight='bold')
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_training_history(history: Dict[str, list], save_path: Optional[Path] = None):
    """Plots training and validation loss & accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2, markersize=4)
    plt.title('Loss vs Epochs (Tamil SER)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy (%)', linewidth=2, markersize=4)
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy (%)', linewidth=2, markersize=4)
    plt.title('Accuracy vs Epochs (Tamil SER)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Path] = None
):
    """Plots confusion matrix with emotion labels."""
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Tamil SER - Confusion Matrix', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_emotion_probabilities(
    probabilities: Dict[str, float],
    title: str = "Tamil Speech Emotion Prediction",
    save_path: Optional[Path] = None
):
    """Plots horizontal bar chart of predicted emotion probabilities."""
    emotions = list(probabilities.keys())
    scores = [probabilities[k] * 100 for k in emotions]
    
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(emotions))
    bars = plt.barh(emotions, scores, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                 ha='left', va='center', fontweight='bold')
                 
    plt.xlim(0, 115)
    plt.xlabel('Confidence (%)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
