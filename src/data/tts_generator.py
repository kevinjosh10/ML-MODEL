import os
import io
from pathlib import Path
from typing import Dict, List
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

# Comprehensive 48 Authentic Tamil Emotional Sentences
TAMIL_EMOTION_PHRASES: Dict[str, List[Dict[str, str]]] = {
    "angry": [
        {"tamil": "போதும் நிறுத்து! இதை என்னால பொறுத்துக்கவே முடியாது!", "trans": "Stop it now! I cannot tolerate this anymore!"},
        {"tamil": "என்ன தைரியம் இருந்தா என்கிட்டயே இப்படி பேசுவ!", "trans": "What audacity you have to speak to me like this!"},
        {"tamil": "உடனே இங்கிருந்து போ! உன் முகத்திலயே முழிக்காத!", "trans": "Get out right now! Don't show your face here!"},
        {"tamil": "நான் சொன்ன வேலையை ஒழுங்கா செய்ய மாட்டியா!", "trans": "Won't you do the work I told you properly!"},
        {"tamil": "என்னை ஏமாத்த பாக்காத, உனக்கு என்ன பைத்தியமா?!", "trans": "Don't try to fool me, are you crazy?!"},
        {"tamil": "இனிமேல் என் விஷயத்துல தலையிடாதே, எச்சரிக்கிறேன்!", "trans": "Don't interfere in my matters anymore, I'm warning you!"},
        {"tamil": "மரியாதையா சொன்னதை செய், சத்தம் போடாதே!", "trans": "Do as I say respectfully, don't make noise!"},
        {"tamil": "சீக்கிரம் வேலைய முடிச்சுட்டு கிளம்பு!", "trans": "Finish the work quickly and leave!"}
    ],
    "happy": [
        {"tamil": "வாவ் சூப்பர்! நாம் வெற்றி பெற்று விட்டோம்!", "trans": "Wow super! We have won!"},
        {"tamil": "இன்னைக்கு எனக்கு ரொம்ப சந்தோஷமான நாள்!", "trans": "Today is such a joyful day for me!"},
        {"tamil": "செம கொண்டாட்டம் இன்னைக்கு, எல்லாரும் வாங்க!", "trans": "Great celebration today, everyone come!"},
        {"tamil": "ரொம்ப நன்றி நண்பா, எனக்கு ரொம்ப பிடிச்சிருக்கு!", "trans": "Thank you so much friend, I really love it!"},
        {"tamil": "இந்த பரிசை நான் எதிர்பார்க்கவே இல்ல, ரொம்ப மகிழ்ச்சி!", "trans": "I didn't expect this gift at all, so happy!"},
        {"tamil": "அருமையான செய்தி, இதை நினைச்சு பெருமையா இருக்கு!", "trans": "Wonderful news, I feel proud thinking of this!"},
        {"tamil": "வாழ்க்கையில இது மறக்க முடியாத அழகான தருணம்!", "trans": "This is an unforgettable beautiful moment in life!"},
        {"tamil": "கனவு நனவாகிடுச்சு, எல்லோருக்கும் என் மனமார்ந்த வாழ்த்துகள்!", "trans": "The dream came true, congratulations to all!"}
    ],
    "sad": [
        {"tamil": "மனசுக்கு ரொம்ப கஷ்டமா இருக்கு... என்ன சொல்றதுன்னே தெரியல...", "trans": "My heart feels so heavy... I don't know what to say..."},
        {"tamil": "என்னை ஏன் எல்லாரும் தனியா விட்டுட்டு போயிட்டீங்க...", "trans": "Why did everyone leave me all alone..."},
        {"tamil": "எல்லாமே போச்சு... இனிமேல் நான் என்ன செய்வேன்...", "trans": "Everything is gone... what will I do now..."},
        {"tamil": "என்னால இந்த வலியை தாங்கிக்கவே முடியல...", "trans": "I cannot bear this pain at all..."},
        {"tamil": "என் வாழ்க்கையில எல்லாமே ஏமாற்றமா தான் முடியுது...", "trans": "Everything in my life ends up in disappointment..."},
        {"tamil": "யாரும் என் பக்கத்துல இல்ல, ரொம்ப தனிமையா உணர்றேன்...", "trans": "Nobody is by my side, I feel so lonely..."},
        {"tamil": "இவ்வளவு தூரம் வந்து இப்படி தோத்து போயிட்டோமே...", "trans": "Coming this far and losing like this..."},
        {"tamil": "எனக்கு இப்போ அழுகை தான் வருது, பேச முடியல...", "trans": "I only feel like crying now, cannot speak..."}
    ],
    "neutral": [
        {"tamil": "வணக்கம், இன்றைய செய்தி அறிக்கையை இப்போது பார்க்கலாம்.", "trans": "Hello, let us look at today's news report now."},
        {"tamil": "நாளை காலை பத்து மணிக்கு அலுவலக கூட்டம் தொடங்கும்.", "trans": "Tomorrow morning at 10 AM the office meeting will start."},
        {"tamil": "சென்னையில் இன்று வானிலை மேகமூட்டத்துடன் காணப்படும்.", "trans": "In Chennai today the weather will be cloudy."},
        {"tamil": "புத்தகம் மேஜையின் மேல் வைக்கப்பட்டிருக்கிறது.", "trans": "The book is placed on top of the table."},
        {"tamil": "அடுத்த ரயில் இன்னும் ஐந்து நிமிடத்தில் வந்து சேரும்.", "trans": "The next train will arrive in 5 minutes."},
        {"tamil": "இந்த ஆவணத்தை சரிபார்த்து கையெழுத்திடுங்கள்.", "trans": "Please verify this document and sign."},
        {"tamil": "தண்ணீர் குடிப்பது உடலுக்கு மிகவும் நல்லது.", "trans": "Drinking water is very good for health."},
        {"tamil": "இன்றைய பாடம் இதோடு முடிவடைகிறது, நன்றி.", "trans": "Today's lesson ends with this, thank you."}
    ],
    "fear": [
        {"tamil": "அங்க ஏதோ விசித்திரமான சத்தம் கேட்குது... எனக்கு பயமா இருக்கு!", "trans": "I hear some strange noise there... I feel so scared!"},
        {"tamil": "யாராவது என்னை காப்பாத்துங்க! உதவி பண்ணுங்க!", "trans": "Someone please save me! Help me!"},
        {"tamil": "இருட்டுல யாரோ நிக்கிற மாதிரி இருக்கு, அங்க போகாதீங்க!", "trans": "Someone seems to be standing in the dark, don't go there!"},
        {"tamil": "ஐயோ! இது என்னன்னு தெரியலையே, கைகள் நடுங்குது!", "trans": "Oh no! I don't know what this is, my hands are shaking!"},
        {"tamil": "கதவை யாரோ தட்டுறாங்க... திறக்க எனக்கு ரொம்ப பயமா இருக்கு!", "trans": "Someone is knocking on the door... I'm terrified to open!"},
        {"tamil": "இங்கிருந்து உடனே ஓடி போயிடுவோம், ஆபத்து!", "trans": "Let us run away from here immediately, danger!"},
        {"tamil": "பயத்துல என் இதயத் துடிப்பு ரொம்ப வேகமா அடிக்குது!", "trans": "My heart is beating so fast in fear!"},
        {"tamil": "என்னை ஒன்னும் பண்ணிடாதீங்க, தயவுசெய்து விட்டுடுங்க!", "trans": "Don't do anything to me, please spare me!"}
    ],
    "surprised": [
        {"tamil": "அப்படியா! நிஜமாவா சொல்றீங்க?! உண்மையிலேயே ஆச்சரியம்!", "trans": "Is it so! Are you serious?! Truly surprising!"},
        {"tamil": "அடடே! இது என்னால நம்பவே முடியலையே, எப்படி சாத்தியம்?!", "trans": "Oh wow! I can hardly believe this, how is it possible?!"},
        {"tamil": "வாவ்! என்ன ஒரு அழகான காட்சி, பிரம்மாண்டமா இருக்கு!", "trans": "Wow! What a beautiful sight, so magnificent!"},
        {"tamil": "என்னது?! முதல் பரிசா?! நிஜமாவா சொல்றீங்க?!", "trans": "What?! First prize?! Are you really serious?!"},
        {"tamil": "இவ்வளவு பெரிய அதிசயத்தை நான் பார்த்ததே இல்ல!", "trans": "I have never seen such a huge wonder!"},
        {"tamil": "திடீர்னு இப்படி நடக்கும்னு நான் கனவுல கூட நினைக்கல!", "trans": "I never even dreamed this would happen all of a sudden!"},
        {"tamil": "அடடா! நீங்களா இது?! என்னால அடையாளமே காண முடியல!", "trans": "Oh my! Is this you?! I couldn't even recognize you!"},
        {"tamil": "இத்தனை நாளா இத மறைச்சு வச்சிருந்தீங்களா, செம சர்ப்ரைஸ்!", "trans": "Were you hiding this all these days, what a surprise!"}
    ]
}

def synthesize_emotional_tts(text: str, emotion: str, sr: int = 16000, target_duration: float = 3.0) -> np.ndarray:
    """
    Synthesizes authentic spoken Tamil audio from text using gTTS
    and applies emotional acoustic prosody morphing (pitch, rate, tremolo, envelope).
    """
    # 1. Render Tamil Text to Speech via gTTS
    tts = gTTS(text=text, lang='ta', slow=(emotion == "sad"))
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # 2. Load into Librosa
    y, orig_sr = librosa.load(mp3_fp, sr=sr, mono=True)
    
    # 3. Apply Emotion-Specific Acoustic Morphing
    if emotion == "angry":
        # Pitch up (+3 semitones), speed up (1.15x), boost energy
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=3.0)
        y = librosa.effects.time_stretch(y, rate=1.15)
        # Apply vocal aggression compression
        y = np.clip(y * 1.6, -1.0, 1.0)
        
    elif emotion == "happy":
        # Upward cheerful pitch (+2.5 semitones), brisk rate (1.1x)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2.5)
        y = librosa.effects.time_stretch(y, rate=1.1)
        # Brightness & vibrato
        t = np.linspace(0, len(y)/sr, len(y))
        vibrato = 1.0 + 0.15 * np.sin(2 * np.pi * 5.0 * t)
        y = y * vibrato
        
    elif emotion == "sad":
        # Downward pitch (-3.0 semitones), slower tempo (0.85x), decay envelope
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3.0)
        y = librosa.effects.time_stretch(y, rate=0.85)
        t = np.linspace(0, len(y)/sr, len(y))
        decay = np.exp(-t * 0.4)
        y = y * decay * 0.75
        
    elif emotion == "fear":
        # Higher pitch (+3.5 semitones), fast rate (1.2x), tremolo jitter
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=3.5)
        y = librosa.effects.time_stretch(y, rate=1.2)
        t = np.linspace(0, len(y)/sr, len(y))
        tremolo = 0.75 + 0.25 * np.sin(2 * np.pi * 12.0 * t)
        y = y * tremolo
        
    elif emotion == "surprised":
        # High pitch peak (+4.5 semitones), fast onset expansion
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=4.5)
        y = librosa.effects.time_stretch(y, rate=1.15)
        
    else:  # neutral
        # Clean standard voice
        pass

    # 4. Standardize Duration to exactly target_duration (3.0s = 48000 samples)
    target_samples = int(sr * target_duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        pad_amount = target_samples - len(y)
        y = np.pad(y, (0, pad_amount), mode='constant')

    # Normalize to 16-bit PCM WAV
    max_val = np.max(np.abs(y)) + 1e-6
    y = y / max_val
    return (y * 32767).astype(np.int16)

def generate_real_tamil_dataset(data_dir: Path, sample_rate: int = 16000, duration: float = 3.0, samples_per_class: int = 8):
    """
    Generates authentic spoken Tamil voice dataset for all 6 emotions using gTTS
    and saves .wav files organized by emotion directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    print("🎙️ Generating authentic spoken Tamil emotional speech dataset with gTTS...")

    for emotion, phrases in TAMIL_EMOTION_PHRASES.items():
        emotion_dir = data_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, item in enumerate(phrases[:samples_per_class]):
            file_name = emotion_dir / f"tamil_{emotion}_{idx+1:03d}.wav"
            if file_name.exists():
                continue
                
            try:
                audio_data = synthesize_emotional_tts(item["tamil"], emotion, sr=sample_rate, target_duration=duration)
                sf.write(str(file_name), audio_data, sample_rate)
                print(f"  • Generated [{emotion.upper()}]: {item['tamil'][:35]}...")
            except Exception as e:
                print(f"  ⚠️ Warning generating {file_name}: {e}")

    print("✅ Tamil emotional speech dataset generation complete!")
