import sys
import os
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.data.vocabulary import tamil_vocab, TamilVocabulary, BLANK_TOKEN
from src.data.tamil_corpus import TAMIL_ASR_CORPUS, synthesize_fallback_tamil_audio
from src.data.dataset import asr_collate_fn
from src.models import TamilASRModel, build_asr_model
from src.training.metrics import calculate_cer, calculate_wer, levenshtein_distance
from src.inference import TamilASRPredictor

def test_vocabulary_encoding_decoding():
    vocab = TamilVocabulary()
    test_text = "வணக்கம் உலகம்"
    
    # Encode
    encoded = vocab.encode(test_text)
    assert len(encoded) == len(test_text)
    assert all(isinstance(x, int) for x in encoded)
    
    # Decode
    decoded = vocab.decode(encoded)
    assert decoded == test_text

def test_ctc_greedy_decoder():
    vocab = TamilVocabulary()
    blank_id = vocab.blank_id
    
    # Target text: "வண"
    v_id = vocab.char_to_id["வ"]
    n_id = vocab.char_to_id["ண"]
    
    # Simulated CTC output with repeated tokens and blanks: [blank, v, v, blank, n, n, blank]
    ctc_sequence = [blank_id, v_id, v_id, blank_id, n_id, n_id, blank_id]
    result = vocab.ctc_greedy_decode(ctc_sequence)
    assert result == "வண"

def test_edit_distance_metrics():
    ref = "வணக்கம்"
    hyp_exact = "வணக்கம்"
    hyp_sub = "வணக்கம்!"
    hyp_diff = "வணககம்"
    
    assert calculate_cer([hyp_exact], [ref]) == 0.0
    assert calculate_wer([hyp_exact], [ref]) == 0.0
    assert calculate_cer([hyp_diff], [ref]) > 0.0

def test_asr_model_forward():
    config = Config()
    config.device = "cpu"
    model = build_asr_model(config)
    
    batch_size = 2
    n_mels = 80
    time_steps = 128
    
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    input_lengths = torch.tensor([time_steps, time_steps // 2], dtype=torch.long)
    
    log_probs, output_lengths = model(dummy_input, input_lengths)
    
    # Time dimension sub-sampled by 4: 128 // 4 = 32
    assert log_probs.shape[0] == batch_size
    assert log_probs.shape[1] == 32
    assert log_probs.shape[2] == config.vocab_size
    assert output_lengths.shape[0] == batch_size
    assert output_lengths[0] == 32
    assert output_lengths[1] == 16

def test_collate_function():
    item1 = {
        "spectrogram": torch.randn(80, 50),
        "input_len": 50,
        "targets": torch.tensor([1, 2, 3], dtype=torch.long),
        "target_len": 3,
        "tamil_text": "வணக்கம்",
        "id": "1"
    }
    item2 = {
        "spectrogram": torch.randn(80, 70),
        "input_len": 70,
        "targets": torch.tensor([4, 5, 6, 7], dtype=torch.long),
        "target_len": 4,
        "tamil_text": "நன்றி",
        "id": "2"
    }
    
    batch = asr_collate_fn([item1, item2])
    assert batch["spectrograms"].shape == (2, 1, 80, 70)
    assert batch["targets"].shape == (2, 4)
    assert torch.equal(batch["input_lengths"], torch.tensor([50, 70]))
    assert torch.equal(batch["target_lengths"], torch.tensor([3, 4]))

def test_asr_inference_predictor():
    config = Config()
    config.device = "cpu"
    model = build_asr_model(config)
    predictor = TamilASRPredictor(model, config=config)
    
    # Synthesize fallback audio
    sample_text = "வணக்கம்"
    audio_data = synthesize_fallback_tamil_audio(sample_text, sr=16000, duration=1.5)
    
    result = predictor.transcribe_waveform(audio_data, sr=16000)
    assert result["status"] == "success"
    assert "tamil_text" in result
    assert "english_translation" in result
    assert result["duration_sec"] > 0
    assert result["words_count"] >= 1

if __name__ == "__main__":
    print("Running Tamil ASR pipeline tests...")
    test_vocabulary_encoding_decoding()
    print("✅ test_vocabulary_encoding_decoding passed")
    test_ctc_greedy_decoder()
    print("✅ test_ctc_greedy_decoder passed")
    test_edit_distance_metrics()
    print("✅ test_edit_distance_metrics passed")
    test_asr_model_forward()
    print("✅ test_asr_model_forward passed")
    test_collate_function()
    print("✅ test_collate_function passed")
    test_asr_inference_predictor()
    print("✅ test_asr_inference_predictor passed")
    print("\n🎉 ALL ASR PIPELINE TESTS PASSED!")
