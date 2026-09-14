# 🎭 Tamil Speech Emotion Recognition (தமிழ் பேச்சு உணர்ச்சி அறிதல்)

A Deep Learning pipeline designed to recognize emotional states from spoken Tamil speech audio clips, optimized for **Google Colab (GPU)** and local environments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_emotion_colab.ipynb)

---

## 🌟 Emotion Categories

| English | தமிழ் (Tamil) | Description |
| :--- | :--- | :--- |
| **Happy** | மகிழ்ச்சி (Magizhchi) | High pitch, elevated energy, positive vocal modulation |
| **Sad** | சோகம் (Sogam) | Lower pitch, slower cadence, low energy |
| **Angry** | கோபம் (Kobam) | High energy, sharp pitch bursts, harsh harmonics |
| **Neutral** | இயல்பு (Iyalbu) | Moderate pitch, flat envelope |
| **Fear** | பயம் (Bayam) | High pitch variations, tremolo modulation |
| **Surprised** | ஆச்சரியம் (Aachariyam) | Peaked vocal pitch, sudden frequency jump |

---

## 🏗️ Model Architecture (Hybrid CNN-BiLSTM-Attention)

1. **Audio Preprocessing**: Converts raw 16 kHz audio waveforms into **Log-Mel Spectrograms** (time $\times$ frequency acoustic energy).
2. **2D CNN Extractor**: Learns spatial frequency patterns and formant shifts across 3 convolutional blocks with Batch Normalization & Dropout.
3. **Bidirectional LSTM**: Captures temporal prosody, inflection, and sentence-level cadence.
4. **Attention Pooling**: Weights key emotional moments in the audio clip.
5. **Classifier Head**: Outputs probability distribution across the 6 emotion classes.

---

## 🚀 Quick Start on Google Colab

1. Open Google Colab and click on **File > Upload Notebook**, then choose `notebooks/tamil_speech_emotion_colab.ipynb` (or use the badge above).
2. Change Runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells sequentially.
4. Use the **Live Recording Cell** to speak Tamil into your microphone and test the emotion prediction on your own voice!

---

## 💻 Local Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Structure
Organize your `.wav` files into emotion folders inside `data/`:
```
data/
├── happy/
│   ├── sample1.wav
│   └── ...
├── sad/
├── angry/
├── neutral/
├── fear/
└── surprised/
```
*(If no audio files exist, the code automatically synthesizes a starter sample dataset to test the full pipeline).*

### 3. Train Model
```bash
python train.py --epochs 25 --batch-size 32 --lr 0.001
```

### 4. Inference on Audio File
```python
from src.inference import TamilSERPredictor

predictor = TamilSERPredictor("checkpoints/best_tamil_ser_model.pth")
result = predictor.predict("path_to_tamil_audio.wav", visualize=True)

print(f"Predicted Emotion: {result['tamil_label']}")
print(f"Confidence: {result['confidence_percentage']}")
```

---

## 📁 Repository Structure
```
ML-MODEL/
├── notebooks/
│   └── tamil_speech_emotion_colab.ipynb  # Interactive Google Colab Notebook
├── src/
│   ├── config.py                         # Configurations & Hyperparameters
│   ├── data/
│   │   ├── audio_preprocessing.py        # Resampling & Log-Mel Spectrogram extraction
│   │   └── dataset.py                    # PyTorch Dataset & Sample Generator
│   ├── models/
│   │   └── ser_model.py                  # Hybrid CNN-BiLSTM-Attention Model
│   ├── training/
│   │   ├── trainer.py                    # GPU AMP Training loop & Checkpointer
│   │   └── metrics.py                    # Evaluation & Confusion Matrix
│   ├── utils/
│   │   ├── audio_recorder.py             # In-browser Colab audio recorder
│   │   └── visualizer.py                 # Waveform & Spectrogram visualizer
│   └── inference.py                      # Single sample & live prediction
├── requirements.txt                      # Project dependencies
├── train.py                              # CLI training script
└── README.md                             # Documentation
```
