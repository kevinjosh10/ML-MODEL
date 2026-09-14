# 🎙️ VOXTAMIL AI: Tamil Speech-to-Text (ASR) Deep Learning Model
### தமிழ் பேச்சு-எழுத்து மாற்றி மாதிரி (Google Colab GPU Optimized)

An end-to-end Deep Learning **Automatic Speech Recognition (ASR)** acoustic model and interactive Web Studio designed to transcribe spoken Tamil audio into Unicode Tamil text characters.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_to_text_colab.ipynb)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success)](https://kevinjosh10.github.io/ML-MODEL/)

---

## 🌟 Key Highlights

- 🧠 **End-to-End Deep Acoustic Model**: 2D Residual-CNN + Multi-layer Bidirectional LSTM + PyTorch Connectionist Temporal Classification (`nn.CTCLoss`).
- 🔤 **Complete Tamil Unicode Vocabulary**: 125 character tokens including உயிர் (vowels), மெய் (consonants), உயிர்மெய் குறிகள் (vowel markers), ஆய்த எழுத்து (ஃ), and whitespace.
- ⚡ **Real-Time Web Studio**: Fast audio waveform visualizer, live microphone dictation (`ta-IN`), audio file upload, text-to-speech audio replay, and Colab GPU backend connectivity.
- 📊 **ASR Metrics Evaluator**: Character Error Rate (CER), Word Error Rate (WER), and Character Accuracy.

---

## 🏗️ Model Architecture

1. **Acoustic Feature Extraction**: Converts raw 16 kHz audio waveforms into 80-channel **Log-Mel Spectrograms** (time $\times$ frequency energy).
2. **2D Residual-CNN Encoder**: Extracts acoustic formants and spectral patterns with $2\times$ temporal sub-sampling.
3. **Multi-layer BiLSTM**: Captures long-range phonetic transitions and conversational Tamil cadence.
4. **CTC Projection Layer**: Linear projection to 125-class Tamil character probabilities with blank token collapsing and greedy decoding.

---

## 🚀 Quick Start on Google Colab

1. Open Google Colab using the badge: **[Open in Google Colab](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_to_text_colab.ipynb)**
2. Ensure GPU is enabled: **Runtime → Change runtime type → T4 GPU → Save**.
3. Click **Runtime → Run all** (`Ctrl + F9`).
4. Follow the live training progress, inspect CER/WER evaluation, record your voice, and launch the public Localtunnel Web Studio!

---

## 💻 Local Setup & Execution

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/kevinjosh10/ML-MODEL.git
cd ML-MODEL
pip install -r requirements.txt
```

### 2. Run Web Studio Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 3. Inference on Audio File
```python
import torch
from src.config import Config
from src.models import build_asr_model
from src.inference import TamilASRPredictor

config = Config()
model = build_asr_model(config)

checkpoint_path = config.checkpoint_dir / "best_tamil_asr_model.pth"
if checkpoint_path.exists():
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state_dict"])

predictor = TamilASRPredictor(model, config)
result = predictor.transcribe_file("data/sample.wav")

print("Transcribed Tamil :", result["tamil_text"])
print("English Meaning   :", result["english_translation"])
print("Confidence Score  :", result["confidence_percentage"])
```

---

## 📁 Repository Structure
```
ML-MODEL/
├── notebooks/
│   ├── tamil_speech_to_text_colab.ipynb   # Turnkey Google Colab Training Notebook (ASR)
│   └── tamil_speech_emotion_colab.ipynb  # Speech Emotion Recognition Notebook
├── src/
│   ├── config.py                          # Configurations & Hyperparameters
│   ├── data/
│   │   ├── vocabulary.py                  # Tamil Unicode Grapheme Tokenizer & CTC Decoder
│   │   ├── tamil_corpus.py                # Tamil sentences & Audio Synthesizer
│   │   ├── audio_preprocessing.py         # Log-Mel Spectrogram Feature Extractor
│   │   └── dataset.py                     # Dynamic Variable-Length PyTorch Dataset & Collate
│   ├── models/
│   │   └── asr_model.py                   # 2D-CNN + BiLSTM + CTC Acoustic Model
│   ├── training/
│   │   ├── trainer.py                     # GPU AMP CTCLoss Trainer
│   │   └── metrics.py                     # Levenshtein Distance CER & WER Evaluator
│   ├── utils/
│   │   └── audio_recorder.py              # In-browser Colab audio recorder
│   └── inference.py                       # Tamil Speech-to-Text inference engine
├── static/
│   ├── css/style.css                      # Modern dark studio stylesheet
│   └── js/app.js                          # Web Audio visualizer, Dictation & API engine
├── templates/
│   └── index.html                         # FastAPI HTML Template
├── index.html                             # GitHub Pages Standalone Web App
├── app.py                                 # FastAPI REST backend
└── requirements.txt                       # Project dependencies
```
