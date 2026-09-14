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
    Inference class for predicting emotions from Tamil speech audio clips.
    """
    def __init__(self, checkpoint_path: Union[str, Path], config: Optional[Config] = None):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.config = checkpoint.get("config", config or Config())
        self.classes = checkpoint.get("classes", self.config.classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = AudioPreprocessor(self.config)

    @torch.no_grad()
    def predict(self, audio_file: Union[str, Path], visualize: bool = True) -> Dict:
        """
        Predicts emotion for an input Tamil audio file.
        Returns predicted emotion (English + Tamil), confidence, and all class probabilities.
        """
        waveform = self.preprocessor.load_audio(audio_file)
        mel_spec = self.preprocessor.extract_mel_spectrogram(waveform)
        
        input_tensor = mel_spec.unsqueeze(0).to(self.device)  # (1, 1, n_mels, time_steps)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        tamil_label = self.config.emotion_map.get(pred_class, pred_class)
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.config.emotion_map.get(cls_name, cls_name): float(probs[i])
            for i, cls_name in enumerate(self.classes)
        }
        
        result = {
            "predicted_class": pred_class,
            "tamil_label": tamil_label,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "probabilities": prob_dict,
            "audio_file": str(audio_file)
        }
        
        if visualize:
            wf_np = waveform.squeeze(0).cpu().numpy()
            spec_np = mel_spec.squeeze(0).cpu().numpy()
            plot_waveform_and_spectrogram(
                wf_np, self.config.sample_rate, spec_np,
                title=f"Prediction: {tamil_label} ({confidence*100:.1f}%)"
            )
            plot_emotion_probabilities(prob_dict, title=f"Emotion Distribution for {Path(audio_file).name}")
            
        return result
