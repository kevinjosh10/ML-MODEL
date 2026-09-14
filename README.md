<div align="center">

# 🎙️ VOXTAMIL AI
### End-to-End Tamil Speech-to-Text (ASR) Acoustic Model & Real-Time Web Studio
**தமிழ் பேச்சு-எழுத்து மாற்றி மாதிரி (Deep Learning ASR)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_to_text_colab.ipynb)
[![GitHub Pages Live](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-00F0FF?style=for-the-badge&logo=githubpages&logoColor=black)](https://kevinjosh10.github.io/ML-MODEL/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Parameters](https://img.shields.io/badge/Parameters-5.00M-F59E0B?style=for-the-badge&logo=target&logoColor=white)](#-model-architecture--parameter-breakdown)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<br/>

[🚀 **Launch Live Web Studio**](https://kevinjosh10.github.io/ML-MODEL/) • [🧠 **Train on Google Colab (GPU)**](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_to_text_colab.ipynb) • [📑 **Model Architecture**](#-model-architecture--parameter-breakdown) • [🔌 **REST API Docs**](#-rest-api-documentation)

---

</div>

## 📌 Executive Overview

**VOXTAMIL AI** is an industrial-grade, end-to-end **Automatic Speech Recognition (ASR)** system engineered specifically for the Tamil language (தமிழ்). Combining **2D Residual Convolutional Neural Networks (Res-CNN)**, **Deep Multi-Layer Bidirectional LSTMs (BiLSTM)**, and **Connectionist Temporal Classification (CTC)**, the system transcribes variable-length spoken Tamil acoustic audio directly into standard Tamil Unicode script with character-level precision.

The repository includes a turnkey **Google Colab GPU Training Pipeline** with mixed-precision acceleration, as well as a full-stack **Interactive Web Studio** deployable to **GitHub Pages** and **FastAPI** with remote Colab GPU bridging.

---

## 🌟 Key Highlights & Engineering Innovations

| Feature | Technical Implementation |
| :--- | :--- |
| 🧠 **Deep Acoustic Model** | 2D Residual-CNN Front-End + 3-Layer BiLSTM + CTC Projection (**4,997,277 Parameters**). |
| 🔤 **Unicode Tokenizer** | 125-token vocabulary mapping base vowels (உயிர்), consonants (மெய்), combining vowel markers (உயிர்மெய் குறிகள்), special symbols (ஃ), and whitespace. |
| ⚡ **Temporal Alignment** | $2\times$ Time Sub-sampling ensuring acoustic frame sequence length $T_{\text{out}} \ge L_{\text{text}}$, eliminating CTC alignment collapse. |
| 🚀 **Mixed-Precision Training** | PyTorch AMP (`torch.cuda.amp.autocast` + `GradScaler`) with AdamW and Cosine Annealing Warm Restarts. |
| 🌐 **Universal Web Studio** | HTML5 Web Audio API visualizer, real-time `ta-IN` Web Speech dictation, audio upload, and Localtunnel remote GPU backend bridge. |
| 📊 **Rigorous Metrics** | Levenshtein dynamic-programming Character Error Rate (CER) and Word Error Rate (WER) evaluators. |

---

## 🏗️ End-to-End System Architecture

```
                                  VOXTAMIL AI ASR PIPELINE
                                  
  ┌────────────────────────┐        ┌─────────────────────────┐        ┌────────────────────────┐
  │  Spoken Tamil Speech   │ ────>  │  Acoustic Preprocessor  │ ────>  │ Log-Mel Spectrogram    │
  │  (16 kHz Mono Audio)   │        │  (1024 FFT, 256 Hop)    │        │ (80 Mel Bins × T Frames│
  └────────────────────────┘        └─────────────────────────┘        └────────────────────────┘
                                                                                    │
  ┌─────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 2D RESIDUAL-CNN ACOUSTIC ENCODER (Sub-sampling: Time / 2, Frequency / 8)             │
│   ├── ConvBlock 1: (1 ──> 32 channels, 3×3 Conv, BatchNorm, GELU, MaxPool 2×2)         │
│   ├── ConvBlock 2: (32 ──> 64 channels, 3×3 Conv, BatchNorm, GELU, MaxPool 2×1)        │
│   └── ConvBlock 3: (64 ──> 128 channels, 3×3 Conv, BatchNorm, GELU, MaxPool 2×1)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │  Flatten Channels × Frequency (128 × 10 = 1280)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. TEMPORAL SEQUENCE ENCODER                                                           │
│   ├── Input Projection: Linear(1280 ──> 256)                                           │
│   ├── 3-Layer Bidirectional LSTM: (Hidden = 256 per dir, Output Dim = 512, Dropout = 0.2)│
│   └── Layer Normalization: LayerNorm(512)                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. CTC DENSE HEAD & DECODER                                                            │
│   ├── Linear(512 ──> 256) ──> GELU ──> Dropout(0.2) ──> Linear(256 ──> 125 Classes)     │
│   ├── PyTorch CTCLoss (with Blank Token ID = 0)                                        │
│   └── Greedy CTC Argmax Decoder (Blank Collapsing & Character Aggregation)             │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                       📝 Transcribed Tamil Text (Unicode)
                       🌐 English Translation & WPM Metrics
```

---

## 📐 Mathematical Formulation

### 1. Connectionist Temporal Classification (CTC) Loss
Given an acoustic input sequence $\mathbf{x} = (x_1, x_2, \dots, x_T)$ and target Tamil character sequence $\mathbf{l} = (l_1, l_2, \dots, l_U)$ where $U \le T$, CTC marginalizes over all valid alignments $\pi \in \mathcal{B}^{-1}(\mathbf{l})$:

$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{l} \mid \mathbf{x}) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x})$$

where $\mathcal{B}$ is the collapse operator that removes consecutive duplicate tokens and blank tokens $\epsilon$.

### 2. Character Error Rate (CER) & Word Error Rate (WER)
Evaluation utilizes dynamic programming Levenshtein edit distance:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%, \quad \text{WER} = \frac{S_w + D_w + I_w}{N_w} \times 100\%$$

where $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, and $N$ = Total reference length.

---

## 📊 Model Architecture & Parameter Breakdown

The complete network contains **4,997,277 learnable parameters** ($pprox 19.99\text{ MB}$ in FP32):

| Layer / Module | Input Shape | Output Shape | Parameters | % of Total |
| :--- | :--- | :--- | :--- | :--- |
| **ConvBlock 1** | $(B, 1, 80, T)$ | $(B, 32, 40, T/2)$ | **9,824** | 0.20% |
| **ConvBlock 2** | $(B, 32, 40, T/2)$ | $(B, 64, 20, T/2)$ | **57,920** | 1.16% |
| **ConvBlock 3** | $(B, 64, 20, T/2)$ | $(B, 128, 10, T/2)$ | **230,528** | 4.61% |
| **Input Linear Projection** | $(B, T/2, 1280)$ | $(B, T/2, 256)$ | **327,936** | 6.56% |
| **BiLSTM Layer 1** | $(B, T/2, 256)$ | $(B, T/2, 512)$ | **1,052,672** | 21.07% |
| **BiLSTM Layer 2** | $(B, T/2, 512)$ | $(B, T/2, 512)$ | **1,576,960** | 31.56% |
| **BiLSTM Layer 3** | $(B, T/2, 512)$ | $(B, T/2, 512)$ | **1,576,960** | 31.56% |
| **Layer Normalization** | $(B, T/2, 512)$ | $(B, T/2, 512)$ | **1,024** | 0.02% |
| **Classification Dense Head** | $(B, T/2, 512)$ | $(B, T/2, 125)$ | **163,453** | 3.27% |
| **TOTAL** | — | — | **4,997,277** | **100.0%** |

---

## 🚀 Quick Start & Reproducibility

### Option A: Google Colab GPU (Recommended • Turnkey)

1. Click the **[Open in Colab](https://colab.research.google.com/github/kevinjosh10/ML-MODEL/blob/main/notebooks/tamil_speech_to_text_colab.ipynb)** badge.
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Press **Runtime → Run all** (`Ctrl + F9`).
4. The notebook will automatically:
   - Clone the repository and install audio dependencies.
   - Synthesize and load the paired Tamil speech-text corpus.
   - Train the 2D-CNN + BiLSTM + CTC model on GPU.
   - Evaluate CER/WER metrics.
   - Test in-notebook microphone recording.
   - Launch a public Localtunnel endpoint for the Web Studio!

---

### Option B: Local CLI Setup & Training

```bash
# 1. Clone repository
git clone https://github.com/kevinjosh10/ML-MODEL.git
cd ML-MODEL

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Train ASR model
python train.py --epochs 35 --batch-size 8 --lr 0.0005
```

---

### Option C: Run Local Web Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 💻 Python Inference Example

```python
import torch
from src.config import Config
from src.models import build_asr_model
from src.inference import TamilASRPredictor

# 1. Initialize configuration & load model checkpoint
config = Config()
model = build_asr_model(config)

checkpoint_path = config.checkpoint_dir / "best_tamil_asr_model.pth"
if checkpoint_path.exists():
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state_dict"])

# 2. Instantiate predictor and transcribe audio
predictor = TamilASRPredictor(model, config)
result = predictor.transcribe_file("data/asr/wavs/sample.wav")

print("=" * 60)
print(f"📝 Tamil Transcript  : {result['tamil_text']}")
print(f"🌐 English Meaning   : {result['english_translation']}")
print(f"✨ Confidence Score  : {result['confidence_percentage']}")
print(f"⚡ Speech Pace       : {result['words_per_minute']} WPM")
print("=" * 60)
```

---

## 🔌 REST API Documentation

The FastAPI backend exposes standard REST endpoints for audio transcription:

### `POST /api/transcribe`
Transcribes an uploaded audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) into Tamil Unicode text.

#### cURL Request:
```bash
curl -X POST "http://localhost:8000/api/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_speech.wav"
```

#### JSON Response:
```json
{
  "status": "success",
  "tamil_text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
  "english_translation": "Hello, how are you?",
  "confidence": 0.984,
  "confidence_percentage": "98.4%",
  "duration_sec": 3.2,
  "words_count": 4,
  "words_per_minute": 75,
  "spectrogram_image": "data:image/png;base64,..."
}
```

---

## 📁 Repository Directory Layout

```
ML-MODEL/
├── notebooks/
│   ├── tamil_speech_to_text_colab.ipynb   # Turnkey GPU Google Colab ASR Training Notebook
│   └── tamil_speech_emotion_colab.ipynb  # Speech Emotion Recognition Pipeline Notebook
├── src/
│   ├── config.py                          # Hyperparameter configuration & path definitions
│   ├── data/
│   │   ├── vocabulary.py                  # 125-token Tamil Unicode grapheme tokenizer & CTC decoder
│   │   ├── tamil_corpus.py                # Authentic Tamil domain corpus & TTS audio generator
│   │   ├── audio_preprocessing.py         # 80-Mel Log-Spectrogram extractor & pre-emphasis filter
│   │   └── dataset.py                     # Variable-length PyTorch TamilASRDataset & collate_fn
│   ├── models/
│   │   └── asr_model.py                   # 2D Res-CNN + BiLSTM + CTC Acoustic Neural Network
│   ├── training/
│   │   ├── trainer.py                     # GPU AMP PyTorch CTCLoss trainer & checkpoint engine
│   │   └── metrics.py                     # Levenshtein distance CER and WER evaluator
│   ├── utils/
│   │   ├── audio_recorder.py              # In-browser Web Audio API PCM WAV recorder for Colab
│   │   └── visualizer.py                  # Waveform & Log-Mel Spectrogram plotting tools
│   └── inference.py                       # Single audio & batch transcription inference engine
├── static/
│   ├── css/styles.css                     # Modern dark studio stylesheet with glassmorphism
│   └── js/app.js                          # Web Audio visualizer, dictation engine, and API client
├── templates/
│   └── index.html                         # FastAPI HTML template interface
├── index.html                             # GitHub Pages standalone client web studio
├── app.py                                 # FastAPI REST server with CORS & Localtunnel support
├── train.py                               # CLI ASR training entrypoint
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Developed by <a href="https://github.com/kevinjosh10"><b>Kevin Joshua</b></a> • Powered by PyTorch & FastAPI</sub>
</div>
