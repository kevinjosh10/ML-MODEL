import torch
from typing import Tuple, List, Dict
from sklearn.metrics import classification_report, accuracy_score
from src.config import Config
from src.utils.visualizer import plot_confusion_matrix

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, test_loader, config: Config) -> Dict:
    """Evaluates the model on test set and computes accuracy, classification report, and confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for specs, labels in test_loader:
        specs = specs.to(config.device)
        outputs = model(specs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())
        
    acc = accuracy_score(all_targets, all_preds)
    
    # Prepare human-readable target names
    target_names = [config.emotion_map.get(cls_name, cls_name) for cls_name in config.classes]
    
    report = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    cm_path = config.output_dir / "tamil_ser_confusion_matrix.png"
    plot_confusion_matrix(all_targets, all_preds, config.classes, save_path=cm_path)
    
    return {
        "accuracy": acc * 100.0,
        "classification_report": report,
        "y_true": all_targets,
        "y_pred": all_preds
    }
