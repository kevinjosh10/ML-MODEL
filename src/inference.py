from pathlib import Path
from typing import Dict, Union, Optional
import torch
import numpy as np
from src.config import Config
from src.data.audio_preprocessing import AudioPreprocessor
from src.models import build_model
from src.utils.visualizer import plot_waveform_and_spectrogram, plot_emotion_probabilities

class TamilSERPredictor:
    """
    Inference class for predicting emotions from Tamil speech audio clips
    using 3-Channel Log-Mel + Delta + Delta-Delta acoustic representations.
    """
    def __init__(self, checkpoint_or_model: Union[str, Path, torch.nn.Module], config: Optional[Config] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(checkpoint_or_model, (str, Path)):
            checkpoint_path = Path(checkpoint_or_model)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            self.config = checkpoint.get("config", config or Config())
            self.classes = checkpoint.get("classes", self.config.classes)
            self.model = build_model(self.config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model = checkpoint_or_model
            self.config = config or Config()
            self.classes = self.config.classes

        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = AudioPreprocessor(self.config, is_train=False)

    @torch.no_grad()
    def predict(self, audio_file: Union[str, Path], visualize: bool = True) -> Dict:
        """
        Predicts emotion for an input Tamil audio file.
        Returns predicted emotion (English + Tamil), confidence, and all class probabilities.
        """
        waveform = self.preprocessor.load_audio(audio_file)
        features = self.preprocessor.extract_mel_spectrogram(waveform, augment=False)  # (3, n_mels, time_steps)
        
        input_tensor = features.unsqueeze(0).to(self.device)  # (1, 3, n_mels, time_steps)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        tamil_label = self.config.emotion_map.get(pred_class, pred_class)
        plot_label = self.config.plot_labels.get(pred_class, pred_class)
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.config.plot_labels.get(cls_name, cls_name): float(probs[i])
            for i, cls_name in enumerate(self.classes)
        }
        
        result = {
            "predicted_class": pred_class,
            "tamil_label": tamil_label,
            "plot_label": plot_label,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "probabilities": prob_dict,
            "audio_file": str(audio_file)
        }
        
        if visualize:
            wf_np = waveform.squeeze().cpu().numpy()
            spec_np = features[0].squeeze().cpu().numpy()  # Channel 0: Log-Mel Spectrogram
            plot_waveform_and_spectrogram(
                wf_np, self.config.sample_rate, spec_np,
                title=f"Predicted: {plot_label} ({confidence*100:.1f}%)"
            )
            plot_emotion_probabilities(prob_dict, title=f"Emotion Distribution: {Path(audio_file).name}")
            
        return result
