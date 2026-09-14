import os
import io
import pytest
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from pathlib import Path
from fastapi.testclient import TestClient

from src.config import Config
from src.models import build_model, TamilSERModel
from src.data.audio_preprocessing import AudioPreprocessor
from src.data.dataset import get_data_loaders, create_sample_dataset
from src.inference import TamilSERPredictor
from app import app

client = TestClient(app)

@pytest.fixture
def temp_config(tmp_path):
    return Config(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs",
        epochs=1,
        batch_size=4,
        device="cpu",
        use_amp=False
    )

def test_model_architecture(temp_config):
    model = build_model(temp_config)
    assert isinstance(model, TamilSERModel)
    
    # Batch of 2 spectrograms: (batch, 1, n_mels, time_steps)
    dummy_input = torch.randn(2, 1, temp_config.n_mels, 94)
    output = model(dummy_input)
    assert output.shape == (2, temp_config.num_classes)

def test_audio_preprocessing(temp_config):
    preprocessor = AudioPreprocessor(temp_config, is_train=False)
    
    # Test waveform length fixing
    short_wave = torch.randn(1, 10000)
    fixed = preprocessor._fix_length(short_wave)
    assert fixed.shape == (1, temp_config.target_samples)
    
    # Test Mel spectrogram extraction
    mel = preprocessor.extract_mel_spectrogram(fixed)
    assert mel.dim() == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == temp_config.n_mels

def test_sample_dataset_generator(temp_config):
    create_sample_dataset(temp_config.data_dir, temp_config.sample_rate, temp_config.duration, samples_per_class=4)
    for emotion in temp_config.classes:
        emotion_dir = temp_config.data_dir / emotion
        assert emotion_dir.exists()
        files = list(emotion_dir.glob("*.wav"))
        assert len(files) == 4

def test_data_loaders(temp_config):
    train_loader, val_loader, test_loader, classes = get_data_loaders(temp_config)
    assert len(classes) == 6
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    specs, labels = next(iter(train_loader))
    assert specs.dim() == 4
    assert specs.shape[1] == 1
    assert specs.shape[2] == temp_config.n_mels

def test_inference_module(temp_config):
    create_sample_dataset(temp_config.data_dir, temp_config.sample_rate, temp_config.duration, samples_per_class=2)
    sample_wav = list((temp_config.data_dir / "happy").glob("*.wav"))[0]
    
    model = build_model(temp_config)
    predictor = TamilSERPredictor(model, config=temp_config)
    result = predictor.predict(sample_wav, visualize=False)
    
    assert "predicted_class" in result
    assert result["predicted_class"] in temp_config.classes
    assert "tamil_label" in result
    assert "confidence" in result
    assert "probabilities" in result

def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data

def test_api_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["presets"]) == 6

def test_api_html_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "தமிழ் உணர்ச்சி AI" in response.text

def test_api_predict_endpoint(temp_config):
    # Generate dummy WAV bytes in memory
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 260 * t) * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, sig)
    buf.seek(0)
    
    response = client.post(
        "/api/predict",
        files={"file": ("test_tamil.wav", buf, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_class" in data
    assert "tamil_name" in data
    assert "confidence" in data
    assert "spectrogram_image" in data
    assert "acoustic_features" in data
