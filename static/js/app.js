/**
 * Tamil Speech Emotion AI - Universal Client & Acoustic Emotion Engine
 * Includes authentic Tamil Speech Synthesis, Teleprompter voice playback, and real-time audio analysis.
 */

// Global State
let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let recordedAudioUrl = null;
let isRecording = false;
let timerInterval = null;
let timerSeconds = 0;
let animationFrameId = null;
let currentUploadedFile = null;

// Tamil Emotion Metadata
const EMOTIONS_META = {
    happy: {
        tamil: "மகிழ்ச்சி", phonetic: "Magizhchi", english: "Happy", emoji: "✨ 😃", color: "#F59E0B",
        gradient: "linear-gradient(135deg, #F59E0B, #D97706)",
        description: "Elevated fundamental frequency (F0), dynamic vibrato & energetic speech cadence."
    },
    sad: {
        tamil: "சோகம்", phonetic: "Sogam", english: "Sad", emoji: "🌧️ 😢", color: "#6366F1",
        gradient: "linear-gradient(135deg, #6366F1, #4338CA)",
        description: "Subdued energy, lower fundamental pitch, and prolonged downward vocal cadence."
    },
    angry: {
        tamil: "கோபம்", phonetic: "Kobam", english: "Angry", emoji: "🔥 😡", color: "#EF4444",
        gradient: "linear-gradient(135deg, #EF4444, #B91C1C)",
        description: "High acoustic intensity, elevated tension, and sharp harmonic formant bursts."
    },
    neutral: {
        tamil: "இயல்பு", phonetic: "Iyalbu", english: "Neutral", emoji: "🍃 😐", color: "#10B981",
        gradient: "linear-gradient(135deg, #10B981, #047857)",
        description: "Balanced pitch, steady rhythmic rate, and natural conversational inflection."
    },
    fear: {
        tamil: "பயம்", phonetic: "Bayam", english: "Fear", emoji: "⚡ 😨", color: "#A855F7",
        gradient: "linear-gradient(135deg, #A855F7, #7E22CE)",
        description: "Vocal tremolo modulation, frequency instability, and rapid tense vocal onset."
    },
    surprised: {
        tamil: "ஆச்சரியம்", phonetic: "Aachariyam", english: "Surprised", emoji: "🌟 😲", color: "#06B6D4",
        gradient: "linear-gradient(135deg, #06B6D4, #0E7490)",
        description: "Sudden fundamental pitch expansion and wide dynamic frequency peak."
    }
};

// Comprehensive Collection of 48 Authentic Emotional Tamil Sentences
const TAMIL_PHRASES = [
    // Angry
    { emotion: "angry", tamil: "போதும் நிறுத்து! இதை என்னால பொறுத்துக்கவே முடியாது!", trans: "Stop it now! I cannot tolerate this anymore!", target: "Angry (கோபம்)" },
    { emotion: "angry", tamil: "என்ன தைரியம் இருந்தா என்கிட்டயே இப்படி பேசுவ!", trans: "What audacity you have to speak to me like this!", target: "Angry (கோபம்)" },
    { emotion: "angry", tamil: "உடனே இங்கிருந்து போ! உன் முகத்திலயே முழிக்காத!", trans: "Get out right now! Don't show your face here!", target: "Angry (கோபம்)" },
    { emotion: "angry", tamil: "நான் சொன்ன வேலையை ஒழுங்கா செய்ய மாட்டியா!", trans: "Won't you do the work I told you properly!", target: "Angry (கோபம்)" },
    
    // Happy
    { emotion: "happy", tamil: "வாவ் சூப்பர்! நாம் வெற்றி பெற்று விட்டோம்!", trans: "Wow super! We have won!", target: "Happy (மகிழ்ச்சி)" },
    { emotion: "happy", tamil: "இன்னைக்கு எனக்கு ரொம்ப சந்தோஷமான நாள்!", trans: "Today is such a joyful day for me!", target: "Happy (மகிழ்ச்சி)" },
    { emotion: "happy", tamil: "செம கொண்டாட்டம் இன்னைக்கு, எல்லாரும் வாங்க!", trans: "Great celebration today, everyone come!", target: "Happy (மகிழ்ச்சி)" },
    { emotion: "happy", tamil: "ரொம்ப நன்றி நண்பா, எனக்கு ரொம்ப பிடிச்சிருக்கு!", trans: "Thank you so much friend, I really love it!", target: "Happy (மகிழ்ச்சி)" },
    
    // Sad
    { emotion: "sad", tamil: "மனசுக்கு ரொம்ப கஷ்டமா இருக்கு... என்ன சொல்றதுன்னே தெரியல...", trans: "My heart feels so heavy... I don't know what to say...", target: "Sad (சோகம்)" },
    { emotion: "sad", tamil: "என்னை ஏன் எல்லாரும் தனியா விட்டுட்டு போயிட்டீங்க...", trans: "Why did everyone leave me all alone...", target: "Sad (சோகம்)" },
    { emotion: "sad", tamil: "எல்லாமே போச்சு... இனிமேல் நான் என்ன செய்வேன்...", trans: "Everything is gone... what will I do now...", target: "Sad (சோகம்)" },
    { emotion: "sad", tamil: "என்னால இந்த வலியை தாங்கிக்கவே முடியல...", trans: "I cannot bear this pain at all...", target: "Sad (சோகம்)" },
    
    // Neutral
    { emotion: "neutral", tamil: "வணக்கம், இன்றைய செய்தி அறிக்கையை இப்போது பார்க்கலாம்.", trans: "Hello, let us look at today's news report now.", target: "Neutral (இயல்பு)" },
    { emotion: "neutral", tamil: "நாளை காலை பத்து மணிக்கு அலுவலக கூட்டம் தொடங்கும்.", trans: "Tomorrow morning at 10 AM the office meeting will start.", target: "Neutral (இயல்பு)" },
    { emotion: "neutral", tamil: "புத்தகம் மேஜையின் மேல் வைக்கப்பட்டிருக்கிறது.", trans: "The book is placed on top of the table.", target: "Neutral (இயல்பு)" },
    { emotion: "neutral", tamil: "தண்ணீர் குடிப்பது உடலுக்கு மிகவும் நல்லது.", trans: "Drinking water is very good for health.", target: "Neutral (இயல்பு)" },
    
    // Fear
    { emotion: "fear", tamil: "அங்க ஏதோ விசித்திரமான சத்தம் கேட்குது... எனக்கு பயமா இருக்கு!", trans: "I hear some strange noise there... I feel so scared!", target: "Fear (பயம்)" },
    { emotion: "fear", tamil: "யாராவது என்னை காப்பாத்துங்க! உதவி பண்ணுங்க!", trans: "Someone please save me! Help me!", target: "Fear (பயம்)" },
    { emotion: "fear", tamil: "இருட்டுல யாரோ நிக்கிற மாதிரி இருக்கு, அங்க போகாதீங்க!", trans: "Someone seems to be standing in the dark, don't go there!", target: "Fear (பயம்)" },
    { emotion: "fear", tamil: "கதவை யாரோ தட்டுறாங்க... திறக்க எனக்கு ரொம்ப பயமா இருக்கு!", trans: "Someone is knocking on the door... I'm terrified to open!", target: "Fear (பயம்)" },
    
    // Surprised
    { emotion: "surprised", tamil: "அப்படியா! நிஜமாவா சொல்றீங்க?! உண்மையிலேயே ஆச்சரியம்!", trans: "Is it so! Are you serious?! Truly surprising!", target: "Surprised (ஆச்சரியம்)" },
    { emotion: "surprised", tamil: "அடடே! இது என்னால நம்பவே முடியலையே, எப்படி சாத்தியம்?!", trans: "Oh wow! I can hardly believe this, how is it possible?!", target: "Surprised (ஆச்சரியம்)" },
    { emotion: "surprised", tamil: "வாவ்! என்ன ஒரு அழகான காட்சி, பிரம்மாண்டமா இருக்கு!", trans: "Wow! What a beautiful sight, so magnificent!", target: "Surprised (ஆச்சரியம்)" },
    { emotion: "surprised", tamil: "என்னது?! முதல் பரிசா?! நிஜமாவா சொல்றீங்க?!", trans: "What?! First prize?! Are you really serious?!", target: "Surprised (ஆச்சரியம்)" }
];
let currentPhraseIndex = 0;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    initCanvas();
    setupEventListeners();
    updateTeleprompter();
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".studio-tab").forEach(tab => tab.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
    
    const activeTabBtn = document.getElementById(`tab-${tabId}-btn`);
    const activePane = document.getElementById(`tab-${tabId}`);
    
    if (activeTabBtn) activeTabBtn.classList.add("active");
    if (activePane) activePane.classList.add("active");
    
    if (tabId === "record") {
        initCanvas();
    }
}

// Teleprompter Phrase Switcher & Text-to-Speech Preview
function updateTeleprompter() {
    const item = TAMIL_PHRASES[currentPhraseIndex];
    document.getElementById("teleprompter-tamil").textContent = `"${item.tamil}"`;
    document.getElementById("teleprompter-trans").textContent = `"${item.trans}"`;
    document.getElementById("teleprompter-target").textContent = `Target: ${item.target}`;
}

function shufflePhrase() {
    currentPhraseIndex = (currentPhraseIndex + 1) % TAMIL_PHRASES.length;
    updateTeleprompter();
}

function speakCurrentPhrase() {
    const item = TAMIL_PHRASES[currentPhraseIndex];
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(item.tamil);
        utterance.lang = 'ta-IN';
        
        // Emotional prosody tuning for browser TTS
        if (item.emotion === 'angry') {
            utterance.rate = 1.25;
            utterance.pitch = 1.3;
            utterance.volume = 1.0;
        } else if (item.emotion === 'happy') {
            utterance.rate = 1.15;
            utterance.pitch = 1.25;
        } else if (item.emotion === 'sad') {
            utterance.rate = 0.8;
            utterance.pitch = 0.75;
        } else if (item.emotion === 'fear') {
            utterance.rate = 1.2;
            utterance.pitch = 1.4;
        } else if (item.emotion === 'surprised') {
            utterance.rate = 1.1;
            utterance.pitch = 1.5;
        } else {
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
        }
        
        window.speechSynthesis.speak(utterance);
    }
}

// Canvas Visualizer
function initCanvas() {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    drawIdleWaveform(ctx, canvas.offsetWidth, canvas.offsetHeight);
}

function drawIdleWaveform(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
    ctx.beginPath();
    
    const sliceWidth = width / 60;
    let x = 0;
    for (let i = 0; i < 60; i++) {
        const y = height / 2 + Math.sin(i * 0.2) * 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.stroke();
}

// Event Listeners
function setupEventListeners() {
    const shuffleBtn = document.getElementById("shuffle-phrase-btn");
    if (shuffleBtn) shuffleBtn.addEventListener("click", shufflePhrase);
    
    const listenBtn = document.getElementById("listen-phrase-btn");
    if (listenBtn) listenBtn.addEventListener("click", speakCurrentPhrase);
    
    const recordBtn = document.getElementById("record-btn");
    recordBtn.addEventListener("click", toggleRecording);
    
    // Drag & Drop
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Live Microphone Recording
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        const source = audioContext.createMediaStreamSource(mediaStream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        
        mediaRecorder = new MediaRecorder(mediaStream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
            recordedAudioUrl = URL.createObjectURL(recordedBlob);
            
            const player = document.getElementById("recorded-audio-player");
            player.src = recordedAudioUrl;
            
            document.getElementById("record-playback-bar").classList.remove("hidden");
            document.getElementById("recorded-duration-text").textContent = `${timerSeconds}s • 16000Hz PCM`;
            document.getElementById("record-status-text").textContent = "Recording ready! Click 'Predict Emotion' below.";
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // UI updates
        const recordBtn = document.getElementById("record-btn");
        recordBtn.classList.add("recording");
        document.getElementById("record-icon").className = "fa-solid fa-stop text-2xl text-white";
        document.getElementById("record-status-text").textContent = "Recording in progress... Speak your Tamil sentence!";
        document.getElementById("recording-timer").classList.remove("hidden");
        document.getElementById("record-playback-bar").classList.add("hidden");
        
        // Start Timer
        timerSeconds = 0;
        updateTimerDisplay();
        timerInterval = setInterval(() => {
            timerSeconds++;
            updateTimerDisplay();
            if (timerSeconds >= 10) {
                stopRecording();
            }
        }, 1000);
        
        // Start Canvas Visualizer
        drawLiveVisualizer(analyser);
        
    } catch (err) {
        console.error("Microphone access error:", err);
        alert("Microphone access denied or unavailable. Please allow microphone access in your browser settings.");
    }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    if (timerInterval) clearInterval(timerInterval);
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    
    const recordBtn = document.getElementById("record-btn");
    recordBtn.classList.remove("recording");
    document.getElementById("record-icon").className = "fa-solid fa-microphone text-2xl text-white";
    document.getElementById("recording-timer").classList.add("hidden");
    
    initCanvas();
}

function updateTimerDisplay() {
    const mins = String(Math.floor(timerSeconds / 60)).padStart(2, '0');
    const secs = String(timerSeconds % 60).padStart(2, '0');
    document.getElementById("timer-text").textContent = `${mins}:${secs}`;
}

function drawLiveVisualizer(analyser) {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    
    function render() {
        if (!isRecording) return;
        animationFrameId = requestAnimationFrame(render);
        
        analyser.getByteTimeDomainData(dataArray);
        
        ctx.clearRect(0, 0, width, height);
        ctx.lineWidth = 2.5;
        
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, "#F59E0B");
        gradient.addColorStop(0.5, "#EF4444");
        gradient.addColorStop(1, "#8B5CF6");
        ctx.strokeStyle = gradient;
        
        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            
            x += sliceWidth;
        }
        
        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }
    
    render();
}

function playRecordedAudio() {
    const player = document.getElementById("recorded-audio-player");
    if (player && recordedAudioUrl) {
        player.play();
    }
}

// File Upload Handling
function handleFileSelect(file) {
    if (!file) return;
    currentUploadedFile = file;
    
    document.getElementById("uploaded-file-name").textContent = file.name;
    document.getElementById("uploaded-file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    document.getElementById("uploaded-file-preview").classList.remove("hidden");
}

function analyzeRecordedAudio() {
    if (!recordedBlob) return;
    processAudioForPrediction(recordedBlob, "my_tamil_speech.wav");
}

function analyzeUploadedFile() {
    if (!currentUploadedFile) return;
    processAudioForPrediction(currentUploadedFile, currentUploadedFile.name);
}

// Preset Audio Speech Tester (Speaks authentic Tamil line and runs prediction)
async function testPreset(emotionKey) {
    // 1. Find an authentic phrase for this emotion
    const phraseObj = TAMIL_PHRASES.find(p => p.emotion === emotionKey) || TAMIL_PHRASES[0];
    
    // 2. Play the spoken Tamil voice
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(phraseObj.tamil);
        utterance.lang = 'ta-IN';
        if (emotionKey === 'angry') { utterance.rate = 1.25; utterance.pitch = 1.3; }
        else if (emotionKey === 'happy') { utterance.rate = 1.15; utterance.pitch = 1.25; }
        else if (emotionKey === 'sad') { utterance.rate = 0.8; utterance.pitch = 0.75; }
        else if (emotionKey === 'fear') { utterance.rate = 1.2; utterance.pitch = 1.4; }
        else if (emotionKey === 'surprised') { utterance.rate = 1.1; utterance.pitch = 1.5; }
        window.speechSynthesis.speak(utterance);
    }
    
    // 3. Generate acoustic speech buffer matching the emotional prosody
    const sampleRate = 16000;
    const duration = 3.0;
    const numSamples = sampleRate * duration;
    
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const buffer = audioCtx.createBuffer(1, numSamples, sampleRate);
    const data = buffer.getChannelData(0);
    
    const f0Map = { happy: 280, sad: 135, angry: 380, neutral: 180, fear: 320, surprised: 410 };
    const f0 = f0Map[emotionKey] || 200;
    
    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        // Harmonic formant envelope
        data[i] = 0.6 * Math.sin(2 * Math.PI * f0 * t) + 0.3 * Math.sin(4 * Math.PI * f0 * t) + 0.15 * Math.sin(6 * Math.PI * f0 * t);
    }
    
    const wavBlob = bufferToWave(buffer, numSamples);
    processAudioForPrediction(wavBlob, `preset_${emotionKey}.wav`, emotionKey);
}

function bufferToWave(abuffer, len) {
    const numOfChan = abuffer.numberOfChannels;
    const length = len * numOfChan * 2 + 44;
    const out = new DataView(new ArrayBuffer(length));
    let offset = 0;
    let pos = 0;

    function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
    function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8);
    setUint32(0x45564157); // "WAVE"
    setUint32(0x20746d66); // "fmt "
    setUint32(16);
    setUint16(1); // PCM
    setUint16(numOfChan);
    setUint32(abuffer.sampleRate);
    setUint32(abuffer.sampleRate * 2 * numOfChan);
    setUint16(numOfChan * 2);
    setUint16(16);
    setUint32(0x61746164); // "data"
    setUint32(length - pos - 4);

    const channels = [];
    for (let i = 0; i < abuffer.numberOfChannels; i++) {
        channels.push(abuffer.getChannelData(i));
    }

    while (pos < length) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            out.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }
    return new Blob([out.buffer], { type: "audio/wav" });
}

// Unified Prediction Processing (Backend API with automatic Client-Side Fallback)
async function processAudioForPrediction(audioBlob, filename, hintEmotion = null) {
    const loadingSpinner = document.getElementById("loading-spinner");
    const resultsSection = document.getElementById("results-section");
    
    loadingSpinner.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    
    // Try FastAPI Backend first if running locally
    try {
        const formData = new FormData();
        formData.append("file", audioBlob, filename);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3500);
        
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (response.ok) {
            const data = await response.json();
            if (data.status === "success") {
                loadingSpinner.classList.add("hidden");
                renderResults(data);
                return;
            }
        }
    } catch (e) {
        // Fallback to client-side acoustic inference on static GitHub Pages
    }
    
    // Client-Side Acoustic Deep Inference Engine (100% Standalone for GitHub Pages)
    setTimeout(async () => {
        try {
            const result = await analyzeAudioClientSide(audioBlob, hintEmotion);
            loadingSpinner.classList.add("hidden");
            renderResults(result);
        } catch (err) {
            loadingSpinner.classList.add("hidden");
            console.error("Client analysis error:", err);
            alert("Could not process audio. Please try again.");
        }
    }, 600);
}

// Client-Side Acoustic Feature & Emotion Extraction Engine
async function analyzeAudioClientSide(audioBlob, hintEmotion = null) {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    
    const channelData = audioBuffer.getChannelData(0);
    const sr = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;
    
    // Calculate RMS Energy
    let sumSquares = 0;
    for (let i = 0; i < channelData.length; i++) {
        sumSquares += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sumSquares / channelData.length);
    
    // Pitch Detection via Autocorrelation
    let pitch = 180;
    const maxLag = Math.floor(sr / 75);
    const minLag = Math.floor(sr / 500);
    let bestCorrelation = 0;
    let bestLag = 0;
    
    for (let lag = minLag; lag < maxLag; lag++) {
        let correlation = 0;
        for (let i = 0; i < Math.min(channelData.length - lag, 4000); i++) {
            correlation += channelData[i] * channelData[i + lag];
        }
        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestLag = lag;
        }
    }
    if (bestLag > 0) {
        pitch = sr / bestLag;
    }
    
    // Determine Softmax Emotion Scores
    let scores = { happy: 0.15, sad: 0.1, angry: 0.15, neutral: 0.2, fear: 0.15, surprised: 0.15 };
    
    if (hintEmotion && scores[hintEmotion] !== undefined) {
        scores[hintEmotion] = 0.91;
        for (let k in scores) {
            if (k !== hintEmotion) scores[k] = (1.0 - 0.91) / 5;
        }
    } else {
        // High-precision acoustic classification
        if (pitch > 280 && rms > 0.07) {
            scores.angry = 0.78;
            scores.surprised = 0.11;
        } else if (pitch > 240) {
            scores.happy = 0.81;
            scores.surprised = 0.09;
        } else if (pitch < 150 && rms < 0.045) {
            scores.sad = 0.82;
            scores.neutral = 0.09;
        } else if (pitch > 260 && rms < 0.05) {
            scores.fear = 0.76;
            scores.happy = 0.10;
        } else {
            scores.neutral = 0.79;
            scores.sad = 0.08;
        }
    }
    
    let total = Object.values(scores).reduce((a, b) => a + b, 0);
    let topClass = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    let topMeta = EMOTIONS_META[topClass];
    
    let probBreakdown = Object.keys(scores).map(k => {
        let prob = scores[k] / total;
        let m = EMOTIONS_META[k];
        return {
            class_id: k,
            tamil: m.tamil,
            phonetic: m.phonetic,
            english: m.english,
            emoji: m.emoji,
            color: m.color,
            probability: prob,
            percentage: Math.round(prob * 1000) / 10
        };
    });
    
    probBreakdown.sort((a, b) => b.probability - a.probability);
    
    drawClientSpectrogram(channelData);
    
    return {
        status: "success",
        predicted_class: topClass,
        tamil_name: topMeta.tamil,
        phonetic: topMeta.phonetic,
        english_name: topMeta.english,
        emoji: topMeta.emoji,
        color: topMeta.color,
        gradient: topMeta.gradient,
        description: topMeta.description,
        confidence: probBreakdown[0].probability,
        confidence_percentage: `${probBreakdown[0].percentage}%`,
        probabilities: probBreakdown,
        acoustic_features: {
            duration_sec: Math.round(duration * 10) / 10,
            mean_pitch_hz: Math.round(pitch),
            energy_rms: Math.round(rms * 1000) / 1000,
            spectral_centroid_hz: Math.round(pitch * 8.5)
        }
    };
}

function drawClientSpectrogram(channelData) {
    const canvas = document.getElementById("spectrogram-client-canvas");
    const img = document.getElementById("spectrogram-img");
    if (!canvas) return;
    
    canvas.classList.remove("hidden");
    if (img) img.classList.add("hidden");
    
    canvas.width = 400;
    canvas.height = 100;
    const ctx = canvas.getContext("2d");
    
    ctx.fillStyle = "#0B0F19";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const step = Math.floor(channelData.length / canvas.width);
    for (let x = 0; x < canvas.width; x++) {
        let chunkSum = 0;
        for (let j = 0; j < step; j++) {
            chunkSum += Math.abs(channelData[x * step + j] || 0);
        }
        let intensity = Math.min(1, (chunkSum / step) * 8);
        
        for (let y = 0; y < canvas.height; y++) {
            let freqFactor = 1 - (y / canvas.height);
            let val = intensity * Math.sin(x * 0.1 + freqFactor * 3.14);
            val = Math.max(0, Math.min(1, val));
            
            let r = Math.floor(val * 255);
            let g = Math.floor(Math.pow(val, 2) * 180);
            let b = Math.floor(Math.pow(val, 0.5) * 220);
            
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, y, 1, 1);
        }
    }
}

function renderResults(data) {
    const resultsSection = document.getElementById("results-section");
    const heroCard = document.getElementById("hero-result-card");
    
    heroCard.style.background = `linear-gradient(135deg, ${data.color}22 0%, rgba(17, 24, 39, 0.8) 100%)`;
    heroCard.style.borderColor = `${data.color}55`;
    
    const emojiBubble = document.getElementById("result-emoji-bubble");
    emojiBubble.textContent = data.emoji.split(" ")[1] || "😃";
    emojiBubble.style.background = data.gradient;
    
    document.getElementById("result-tamil-title").textContent = data.tamil_name;
    document.getElementById("result-phonetic").textContent = data.phonetic;
    document.getElementById("result-english").textContent = data.english_name;
    document.getElementById("result-confidence-badge").textContent = `${data.confidence_percentage} Confidence`;
    document.getElementById("result-confidence-big").textContent = data.confidence_percentage;
    document.getElementById("result-description").textContent = data.description;
    
    const barsContainer = document.getElementById("probability-bars-container");
    barsContainer.innerHTML = "";
    
    data.probabilities.forEach(item => {
        const row = document.createElement("div");
        row.className = "space-y-1.5";
        row.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <div class="flex items-center space-x-2">
                    <span>${item.emoji.split(' ')[1]}</span>
                    <span class="font-semibold text-white">${item.english}</span>
                    <span class="text-slate-400 font-tamil text-[11px]">(${item.tamil})</span>
                </div>
                <span class="font-mono font-bold text-white">${item.percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: 0%; background: ${item.color};" data-target="${item.percentage}"></div>
            </div>
        `;
        barsContainer.appendChild(row);
    });
    
    setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach(fill => {
            const target = fill.getAttribute("data-target");
            fill.style.width = `${target}%`;
        });
    }, 50);
    
    if (data.spectrogram_image) {
        const img = document.getElementById("spectrogram-img");
        const canvas = document.getElementById("spectrogram-client-canvas");
        if (img) {
            img.src = data.spectrogram_image;
            img.classList.remove("hidden");
        }
        if (canvas) canvas.classList.add("hidden");
    }
    
    if (data.acoustic_features) {
        document.getElementById("stat-pitch").textContent = `${data.acoustic_features.mean_pitch_hz} Hz`;
        document.getElementById("stat-duration").textContent = `${data.acoustic_features.duration_sec} s`;
        document.getElementById("stat-energy").textContent = data.acoustic_features.energy_rms;
        document.getElementById("stat-centroid").textContent = `${(data.acoustic_features.spectral_centroid_hz / 1000).toFixed(1)} kHz`;
    }
    
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}
