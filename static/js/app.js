/**
 * Tamil Speech Emotion AI - Frontend Application
 * Interacts with Web Audio API, Canvas Visualizer, MediaRecorder, and FastAPI Backend.
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

// Tamil Practice Phrases
const TAMIL_PHRASES = [
    {
        emotion: "happy",
        tamil: "எனக்கு ரொம்ப சந்தோஷமா இருக்கு, வெற்றி பெற்று விட்டோம்!",
        trans: "I am so happy, we have won!",
        target: "மகிழ்ச்சி (Happy)"
    },
    {
        emotion: "sad",
        tamil: "மனசுக்கு ரொம்ப கஷ்டமா இருக்கு, என்ன சொல்றதுன்னே தெரியல.",
        trans: "My heart feels very heavy, I don't know what to say.",
        target: "சோகம் (Sad)"
    },
    {
        emotion: "angry",
        tamil: "இதை என்னால பொறுத்துக்கவே முடியாது, உடனே நிறுத்துங்கள்!",
        trans: "I cannot tolerate this anymore, stop it right now!",
        target: "கோபம் (Angry)"
    },
    {
        emotion: "neutral",
        tamil: "வணக்கம், இன்றைய செய்தி அறிக்கையை இப்போது பார்க்கலாம்.",
        trans: "Hello, let us look at today's news report now.",
        target: "இயல்பு (Neutral)"
    },
    {
        emotion: "fear",
        tamil: "அங்க ஏதோ விசித்திரமான சத்தம் கேட்குது, எனக்கு பயமா இருக்கு!",
        trans: "I hear some strange noise there, I feel scared!",
        target: "பயம் (Fear)"
    },
    {
        emotion: "surprised",
        tamil: "அப்படியா! இதை என்னால நம்பவே முடியல, உண்மையிலேயே ஆச்சரியம்!",
        trans: "Is it so! I can hardly believe it, truly surprising!",
        target: "ஆச்சரியம் (Surprised)"
    }
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

// Teleprompter Phrase Switcher
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

// Canvas Visualizer
function initCanvas() {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    // Set actual pixel dimensions
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
    document.getElementById("shuffle-phrase-btn").addEventListener("click", shufflePhrase);
    
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
    
    if (timerInterval) {
        clearInterval(timerInterval);
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
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
    const formData = new FormData();
    formData.append("file", recordedBlob, "my_tamil_speech.wav");
    sendPredictionRequest(formData);
}

function analyzeUploadedFile() {
    if (!currentUploadedFile) return;
    const formData = new FormData();
    formData.append("file", currentUploadedFile, currentUploadedFile.name);
    sendPredictionRequest(formData);
}

// Preset Audio Generator & Tester
async function testPreset(emotionKey) {
    // Generate synthetic audio sample on the client and send for inference
    const sampleRate = 16000;
    const duration = 3.0;
    const numSamples = sampleRate * duration;
    
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const buffer = audioCtx.createBuffer(1, numSamples, sampleRate);
    const data = buffer.getChannelData(0);
    
    // Tone simulation based on emotion
    const f0Map = { happy: 260, sad: 130, angry: 340, neutral: 175, fear: 290, surprised: 360 };
    const f0 = f0Map[emotionKey] || 200;
    
    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        data[i] = 0.6 * Math.sin(2 * Math.PI * f0 * t) + 0.3 * Math.sin(4 * Math.PI * f0 * t);
    }
    
    // Convert buffer to WAV Blob
    const wavBlob = bufferToWave(buffer, numSamples);
    const formData = new FormData();
    formData.append("file", wavBlob, `preset_${emotionKey}.wav`);
    sendPredictionRequest(formData);
}

function bufferToWave(abuffer, len) {
    const numOfChan = abuffer.numberOfChannels;
    const length = len * numOfChan * 2 + 44;
    const out = new DataView(new ArrayBuffer(length));
    const channels = [];
    let sample = 0;
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

    for (let i = 0; i < abuffer.numberOfChannels; i++) {
        channels.push(abuffer.getChannelData(i));
    }

    while (pos < length) {
        for (let i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            out.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }
    return new Blob([out.buffer], { type: "audio/wav" });
}

// Backend API Request & Result Rendering
async function sendPredictionRequest(formData) {
    const loadingSpinner = document.getElementById("loading-spinner");
    const resultsSection = document.getElementById("results-section");
    
    loadingSpinner.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        loadingSpinner.classList.add("hidden");
        
        if (data.status === "success") {
            renderResults(data);
        } else {
            alert(`Prediction Error: ${data.message || 'Failed to process audio'}`);
        }
    } catch (err) {
        loadingSpinner.classList.add("hidden");
        console.error("API error:", err);
        alert("Failed to connect to the prediction server. Make sure the backend is running.");
    }
}

function renderResults(data) {
    const resultsSection = document.getElementById("results-section");
    const heroCard = document.getElementById("hero-result-card");
    
    // Update Hero Card styling & colors
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
    
    // Update Probability Bars
    const barsContainer = document.getElementById("probability-bars-container");
    barsContainer.innerHTML = "";
    
    data.probabilities.forEach(item => {
        const row = document.createElement("div");
        row.className = "space-y-1.5";
        row.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <div class="flex items-center space-x-2">
                    <span>${item.emoji.split(' ')[1]}</span>
                    <span class="font-tamil font-semibold text-white">${item.tamil}</span>
                    <span class="text-slate-400 font-mono text-[11px]">(${item.english})</span>
                </div>
                <span class="font-mono font-bold text-white">${item.percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: 0%; background: ${item.color};" data-target="${item.percentage}"></div>
            </div>
        `;
        barsContainer.appendChild(row);
    });
    
    // Animate Bar Fills
    setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach(fill => {
            const target = fill.getAttribute("data-target");
            fill.style.width = `${target}%`;
        });
    }, 50);
    
    // Update Spectrogram & Acoustic Stats
    if (data.spectrogram_image) {
        document.getElementById("spectrogram-img").src = data.spectrogram_image;
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
