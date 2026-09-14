import os
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.config import Config
from src.data.tamil_corpus import TAMIL_ASR_CORPUS, generate_tamil_asr_dataset
from src.data.vocabulary import tamil_vocab
from src.models import build_asr_model
from src.inference import TamilASRPredictor

# Initialize app and templates
app = FastAPI(title="Tamil Speech-to-Text AI")

# Enable Cross-Origin Resource Sharing (CORS) for external clients & GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data" / "asr"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global configuration
config = Config()

# Global predictor instance
asr_predictor: Optional[TamilASRPredictor] = None

def get_or_create_predictor() -> TamilASRPredictor:
    global asr_predictor
    if asr_predictor is not None:
        return asr_predictor
        
    best_ckpt = CHECKPOINT_DIR / "best_tamil_asr_model.pth"
    if best_ckpt.exists():
        asr_predictor = TamilASRPredictor(best_ckpt, config=config)
    else:
        model = build_asr_model(config)
        asr_predictor = TamilASRPredictor(model, config=config)
        
    return asr_predictor

def generate_spectrogram_base64(waveform: np.ndarray, sr: int) -> str:
    """Generates an aesthetic Log-Mel Spectrogram image returned as base64 string."""
    plt.figure(figsize=(7, 2.2), facecolor='#0B0F19')
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#0B0F19')
    
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    librosa.display.specshow(log_mel, sr=sr, hop_length=256, x_axis=None, y_axis=None, cmap='magma', ax=ax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=180, facecolor='#0B0F19')
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main Web Studio UI."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_corpus": TAMIL_ASR_CORPUS[:8],
        "device": config.device.upper()
    })

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receives audio file or microphone recording and returns transcribed Tamil text."""
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")
            
        try:
            audio_io = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_io)
        except Exception:
            audio_io.seek(0)
            y, sr = librosa.load(audio_io, sr=16000, mono=False)
            
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
            
        predictor = get_or_create_predictor()
        result = predictor.transcribe_waveform(y, sr)
        
        # Add visual spectrogram
        result["spectrogram_image"] = generate_spectrogram_base64(y, sr)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/samples")
async def get_sample_library():
    """Returns curated Tamil speech sample corpus."""
    return JSONResponse(content={
        "status": "success",
        "samples": TAMIL_ASR_CORPUS
    })

@app.get("/api/health")
async def health_check():
    """Health status and model information."""
    return {
        "status": "online",
        "task": "Tamil Speech-to-Text (ASR)",
        "model": "Residual-CNN + BiLSTM + CTC",
        "sample_rate": config.sample_rate,
        "device": config.device,
        "vocab_size": config.vocab_size
    }
