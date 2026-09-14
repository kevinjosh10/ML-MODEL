import os
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import Config
from src.data.dataset import create_sample_dataset
from src.data.audio_preprocessing import AudioPreprocessor
from src.models import build_model
from src.inference import TamilSERPredictor

# Initialize app and templates
app = FastAPI(title="Tamil Speech Emotion AI")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global configuration & emotion metadata
config = Config()
EMOTION_META = {
    "happy": {
        "tamil": "மகிழ்ச்சி",
        "phonetic": "Magizhchi",
        "english": "Happy",
        "emoji": "✨ 😃",
        "color": "#F59E0B",
        "gradient": "linear-gradient(135deg, #F59E0B, #D97706)",
        "description": "Elevated fundamental frequency (F0), dynamic vibrato & energetic speech cadence.",
        "sample_phrase": "வாவ் சூப்பர், நாம் வெற்றி பெற்று விட்டோம்!",
        "phrase_trans": "Wow super, we have won!"
    },
    "sad": {
        "tamil": "சோகம்",
        "phonetic": "Sogam",
        "english": "Sad",
        "emoji": "🌧️ 😢",
        "color": "#6366F1",
        "gradient": "linear-gradient(135deg, #6366F1, #4338CA)",
        "description": "Subdued energy, lower fundamental pitch, and prolonged downward vocal cadence.",
        "sample_phrase": "மனசுக்கு ரொம்ப கஷ்டமா இருக்கு, என்ன சொல்றதுன்னே தெரியல...",
        "phrase_trans": "My heart feels very heavy, I don't know what to say..."
    },
    "angry": {
        "tamil": "கோபம்",
        "phonetic": "Kobam",
        "english": "Angry",
        "emoji": "🔥 😡",
        "color": "#EF4444",
        "gradient": "linear-gradient(135deg, #EF4444, #B91C1C)",
        "description": "High acoustic intensity, elevated tension, and sharp harmonic formant bursts.",
        "sample_phrase": "போதும் நிறுத்து! இதை என்னால பொறுத்துக்கவே முடியாது!",
        "phrase_trans": "Stop it now! I cannot tolerate this anymore!"
    },
    "neutral": {
        "tamil": "இயல்பு",
        "phonetic": "Iyalbu",
        "english": "Neutral",
        "emoji": "🍃 😐",
        "color": "#10B981",
        "gradient": "linear-gradient(135deg, #10B981, #047857)",
        "description": "Balanced pitch, steady rhythmic rate, and natural conversational inflection.",
        "sample_phrase": "வணக்கம், இன்றைய செய்தி அறிக்கையை இப்போது பார்க்கலாம்.",
        "phrase_trans": "Hello, let us look at today's news report now."
    },
    "fear": {
        "tamil": "பயம்",
        "phonetic": "Bayam",
        "english": "Fear",
        "emoji": "⚡ 😨",
        "color": "#A855F7",
        "gradient": "linear-gradient(135deg, #A855F7, #7E22CE)",
        "description": "Vocal tremolo modulation, frequency instability, and rapid tense vocal onset.",
        "sample_phrase": "அங்க ஏதோ விசித்திரமான சத்தம் கேட்குது... எனக்கு ரொம்ப பயமா இருக்கு!",
        "phrase_trans": "I hear some strange noise there... I feel so scared!"
    },
    "surprised": {
        "tamil": "ஆச்சரியம்",
        "phonetic": "Aachariyam",
        "english": "Surprised",
        "emoji": "🌟 😲",
        "color": "#06B6D4",
        "gradient": "linear-gradient(135deg, #06B6D4, #0E7490)",
        "description": "Sudden fundamental pitch expansion and wide dynamic frequency peak.",
        "sample_phrase": "அப்படியா! நிஜமாவா சொல்றீங்க?! உண்மையிலேயே ஆச்சரியம்!",
        "phrase_trans": "Is it so! Are you serious?! Truly surprising!"
    }
}

# Global predictor instance
predictor_instance: Optional[TamilSERPredictor] = None

def get_or_create_predictor() -> TamilSERPredictor:
    global predictor_instance
    if predictor_instance is not None:
        return predictor_instance
        
    best_ckpt = CHECKPOINT_DIR / "best_tamil_ser_model.pth"
    if not best_ckpt.exists():
        latest_ckpt = CHECKPOINT_DIR / "latest_tamil_ser_model.pth"
        if latest_ckpt.exists():
            best_ckpt = latest_ckpt
            
    if best_ckpt.exists():
        predictor_instance = TamilSERPredictor(best_ckpt, config=config)
    else:
        create_sample_dataset(DATA_DIR, config.sample_rate, config.duration, samples_per_class=30)
        model = build_model(config)
        predictor_instance = TamilSERPredictor(model, config=config)
        
    return predictor_instance

def generate_spectrogram_base64(waveform: np.ndarray, sr: int) -> str:
    """Generates an aesthetic Log-Mel Spectrogram image returned as base64 string."""
    plt.figure(figsize=(7, 2.2), facecolor='#0B0F19')
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#0B0F19')
    
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis=None, y_axis=None, cmap='magma', ax=ax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=180, facecolor='#0B0F19')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Calculates acoustic features: pitch, energy, spectral centroid."""
    duration = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y**2)))
    
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=500)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 180.0
    except Exception:
        mean_pitch = 180.0
        
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_cent = float(np.mean(cent))
    except Exception:
        mean_cent = 2200.0
        
    return {
        "duration_sec": round(duration, 2),
        "mean_pitch_hz": round(mean_pitch, 1),
        "energy_rms": round(rms, 4),
        "spectral_centroid_hz": round(mean_cent, 1)
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main Web App UI."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotions": EMOTION_META,
        "classes": config.classes,
        "device": config.device.upper()
    })

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Receives audio file or microphone recording and returns predicted emotion."""
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")
            
        try:
            audio_io = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_io)
        except Exception:
            audio_io.seek(0)
            y, sr = librosa.load(audio_io, sr=None, mono=False)
            
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
            
        if sr != config.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate)
            sr = config.sample_rate

        y = y.astype(np.float32)
        max_val = np.max(np.abs(y)) + 1e-6
        if max_val > 1.0:
            y = y / max_val
            
        waveform_tensor = torch.from_numpy(y).unsqueeze(0)
        preprocessor = AudioPreprocessor(config, is_train=False)
        waveform_tensor = preprocessor._fix_length(waveform_tensor)
        features = preprocessor.extract_mel_spectrogram(waveform_tensor, augment=False)  # (3, n_mels, time_steps)
        
        predictor = get_or_create_predictor()
        input_tensor = features.unsqueeze(0).to(predictor.device)
        
        with torch.no_grad():
            logits = predictor.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        pred_class = config.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        meta = EMOTION_META[pred_class]
        
        prob_breakdown = []
        for i, cls_name in enumerate(config.classes):
            c_meta = EMOTION_META[cls_name]
            p_val = float(probs[i])
            prob_breakdown.append({
                "class_id": cls_name,
                "tamil": c_meta["tamil"],
                "phonetic": c_meta["phonetic"],
                "english": c_meta["english"],
                "emoji": c_meta["emoji"],
                "color": c_meta["color"],
                "probability": round(p_val, 4),
                "percentage": round(p_val * 100, 1)
            })
            
        prob_breakdown.sort(key=lambda x: x["probability"], reverse=True)
        
        acoustic_info = extract_audio_features(y, sr)
        spectrogram_b64 = generate_spectrogram_base64(y, sr)
        
        return JSONResponse({
            "status": "success",
            "predicted_class": pred_class,
            "tamil_name": meta["tamil"],
            "phonetic": meta["phonetic"],
            "english_name": meta["english"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "gradient": meta["gradient"],
            "description": meta["description"],
            "sample_phrase": meta["sample_phrase"],
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "probabilities": prob_breakdown,
            "acoustic_features": acoustic_info,
            "spectrogram_image": f"data:image/png;base64,{spectrogram_b64}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/presets")
async def get_presets():
    """Returns sample Tamil emotional voice presets to test immediately."""
    presets = []
    for cls_name, meta in EMOTION_META.items():
        presets.append({
            "class_id": cls_name,
            "tamil_name": meta["tamil"],
            "phonetic": meta["phonetic"],
            "english_name": meta["english"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "sample_phrase": meta["sample_phrase"],
            "phrase_trans": meta["phrase_trans"]
        })
    return JSONResponse({"status": "success", "presets": presets})

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Tamil SER 3-Channel CNN-BiLSTM-Attention",
        "device": config.device.upper(),
        "classes": config.classes
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Tamil Speech Emotion AI Web Server on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
