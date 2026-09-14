/**
 * VOXTAMIL AI - Neural Brain & Architecture Visualizer Engine
 * Simulates the biological & deep learning neural network topology of the 
 * 2D Residual-CNN + 3-Layer BiLSTM + CTC Acoustic Speech Recognition Model (4.99M Parameters).
 */

class NeuralBrainVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.neurons = [];
        this.synapses = [];
        this.particles = [];
        this.viewMode = 'brain'; // 'brain' | 'layers'
        this.morphProgress = 0.0; // 0 = brain, 1 = layers
        this.targetMorph = 0.0;
        this.hoveredNeuron = null;
        this.audioStimulation = 0.0; // 0.0 to 1.0 based on live audio
        this.mouseX = -1000;
        this.mouseY = -1000;
        this.pulseBurstTimer = 0;
        this.animationId = null;

        // Model Layer Configurations
        this.layerMeta = [
            { id: 0, name: "Log-Mel Spectrogram", type: "Input", color: "#F59E0B", glow: "rgba(245, 158, 11, 0.8)", desc: "80 Mel filterbank acoustic frequency bins extracted from 16kHz speech.", tensor: "[Batch, 1, 80, Time]", params: "0" },
            { id: 1, name: "Res-CNN ConvBlock 1", type: "Conv2D", color: "#FBBF24", glow: "rgba(251, 191, 36, 0.8)", desc: "32 filters, 3×3 conv, BatchNorm, GELU, 2×2 MaxPool (Time/2, Freq/2).", tensor: "[Batch, 32, 40, Time/2]", params: "9,824" },
            { id: 2, name: "Res-CNN ConvBlock 2 & 3", type: "Conv2D", color: "#06B6D4", glow: "rgba(6, 182, 212, 0.8)", desc: "64 to 128 deep feature maps, residual connections, Freq/8 pooling.", tensor: "[Batch, 128, 10, Time/2]", params: "288,448" },
            { id: 3, name: "Feature Bottleneck Projection", type: "Linear", color: "#38BDF8", glow: "rgba(56, 189, 248, 0.8)", desc: "Linear projection flattens 128 channels × 10 freq = 1280 into 256.", tensor: "[Batch, Time/2, 256]", params: "327,936" },
            { id: 4, name: "BiLSTM Layer 1 & 2", type: "Recurrent", color: "#6366F1", glow: "rgba(99, 102, 241, 0.8)", desc: "Forward & backward phonetic recurrence capturing Tamil vowel formants.", tensor: "[Batch, Time/2, 512]", params: "2,629,632" },
            { id: 5, name: "BiLSTM Layer 3 & LayerNorm", type: "Recurrent", color: "#818CF8", glow: "rgba(129, 140, 248, 0.8)", desc: "Deep temporal contextualization with LayerNorm stability.", tensor: "[Batch, Time/2, 512]", params: "1,577,984" },
            { id: 6, name: "CTC Dense Projection", type: "Dense/CTC", color: "#10B981", glow: "rgba(16, 185, 129, 0.8)", desc: "Outputs posterior character probabilities across 125 Tamil Unicode graphemes.", tensor: "[Batch, Time/2, 125]", params: "163,453" }
        ];

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.createNetwork();
        this.setupInteractions();
        this.startLoop();
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.getBoundingClientRect();
        this.width = this.canvas.width = rect.width * (window.devicePixelRatio || 1);
        this.height = this.canvas.height = rect.height * (window.devicePixelRatio || 1);
        this.scale = window.devicePixelRatio || 1;
        if (this.neurons.length > 0) {
            this.recalculatePositions();
        }
    }

    createNetwork() {
        this.neurons = [];
        this.synapses = [];
        this.particles = [];

        const layerCounts = [14, 18, 22, 16, 24, 20, 16]; // ~130 representative neuron clusters
        let globalId = 0;

        // 1. Generate Neurons
        layerCounts.forEach((count, layerIdx) => {
            const meta = this.layerMeta[layerIdx];
            for (let i = 0; i < count; i++) {
                const fraction = count > 1 ? i / (count - 1) : 0.5;

                // --- LAYERED TOPOLOGY (Column X, Spread Y) ---
                const padX = this.width * 0.08;
                const padY = this.height * 0.14;
                const colW = (this.width - padX * 2) / (this.layerMeta.length - 1);
                const rowH = (this.height - padY * 2);

                const lx = padX + layerIdx * colW + (Math.random() - 0.5) * 8 * this.scale;
                const ly = padY + fraction * rowH + (Math.sin(layerIdx * 1.5 + i) * 12) * this.scale;

                // --- ORGANIC BRAIN TOPOLOGY (Anatomical Dual-Hemisphere) ---
                const centerX = this.width * 0.5;
                const centerY = this.height * 0.48;
                const brainRadiusX = this.width * 0.36;
                const brainRadiusY = this.height * 0.34;

                const u = (layerIdx / (this.layerMeta.length - 1)) * Math.PI - Math.PI / 2; // -PI/2 to +PI/2
                const hemisphere = i % 2 === 0 ? 1 : -1;
                const spread = (Math.random() * 0.25 + 0.75);

                const angle = u + (fraction - 0.5) * 0.85;
                const rX = brainRadiusX * spread * (0.85 + Math.cos(angle * 2) * 0.15);
                const rY = brainRadiusY * spread * (0.85 + Math.sin(angle) * 0.15);

                const bx = centerX + hemisphere * Math.abs(Math.cos(angle)) * rX * 0.88 + (hemisphere * 18 * this.scale);
                const by = centerY + Math.sin(angle) * rY + (Math.sin(i * 3) * 15 * this.scale);

                this.neurons.push({
                    id: globalId++,
                    layerIdx: layerIdx,
                    meta: meta,
                    lx: lx,
                    ly: ly,
                    bx: bx,
                    by: by,
                    x: bx, // current position
                    y: by,
                    baseRadius: (3.5 + Math.random() * 2.5) * this.scale,
                    radius: 4 * this.scale,
                    energy: Math.random(),
                    pulsePhase: Math.random() * Math.PI * 2,
                    tamilGrapheme: layerIdx === 6 ? this.getTamilSampleChar(i) : null
                });
            }
        });

        // 2. Generate Synapses (Axons) between adjacent and recurrent layers
        this.neurons.forEach(n1 => {
            // Forward connections to next layer
            const nextLayerNeurons = this.neurons.filter(n => n.layerIdx === n1.layerIdx + 1);
            if (nextLayerNeurons.length > 0) {
                const connectionCount = Math.min(3, nextLayerNeurons.length);
                const shuffled = [...nextLayerNeurons].sort(() => Math.random() - 0.5);
                for (let k = 0; k < connectionCount; k++) {
                    const n2 = shuffled[k];
                    this.synapses.push({
                        from: n1,
                        to: n2,
                        weight: 0.3 + Math.random() * 0.7,
                        color: n1.meta.color,
                        isRecurrent: false
                    });
                }
            }

            // BiLSTM Lateral / Recurrent Synapses within Layer 4 and 5
            if (n1.layerIdx === 4 || n1.layerIdx === 5) {
                const sameLayer = this.neurons.filter(n => n.layerIdx === n1.layerIdx && n.id !== n1.id);
                if (sameLayer.length > 0 && Math.random() > 0.65) {
                    const n2 = sameLayer[Math.floor(Math.random() * sameLayer.length)];
                    this.synapses.push({
                        from: n1,
                        to: n2,
                        weight: 0.85,
                        color: "#818CF8",
                        isRecurrent: true
                    });
                }
            }
        });

        // 3. Action Potential Particles
        for (let p = 0; p < 80; p++) {
            const syn = this.synapses[Math.floor(Math.random() * this.synapses.length)];
            this.particles.push({
                synapse: syn,
                progress: Math.random(),
                speed: 0.006 + Math.random() * 0.012,
                size: (1.5 + Math.random() * 1.5) * this.scale,
                color: syn.color
            });
        }
    }

    getTamilSampleChar(index) {
        const chars = ["அ", "ஆ", "இ", "ஈ", "உ", "க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ா", "ி", "ீ", "ு", "ூ", "்", "ஃ", " "];
        return chars[index % chars.length];
    }

    recalculatePositions() {
        const layerCounts = [14, 18, 22, 16, 24, 20, 16];
        let nIdx = 0;

        layerCounts.forEach((count, layerIdx) => {
            for (let i = 0; i < count; i++) {
                if (nIdx >= this.neurons.length) break;
                const n = this.neurons[nIdx++];
                const fraction = count > 1 ? i / (count - 1) : 0.5;

                const padX = this.width * 0.08;
                const padY = this.height * 0.14;
                const colW = (this.width - padX * 2) / (this.layerMeta.length - 1);
                const rowH = (this.height - padY * 2);

                n.lx = padX + layerIdx * colW;
                n.ly = padY + fraction * rowH;

                const centerX = this.width * 0.5;
                const centerY = this.height * 0.48;
                const brainRadiusX = this.width * 0.36;
                const brainRadiusY = this.height * 0.34;

                const u = (layerIdx / (this.layerMeta.length - 1)) * Math.PI - Math.PI / 2;
                const hemisphere = i % 2 === 0 ? 1 : -1;
                const angle = u + (fraction - 0.5) * 0.85;
                const rX = brainRadiusX * (0.85 + Math.cos(angle * 2) * 0.15);
                const rY = brainRadiusY * (0.85 + Math.sin(angle) * 0.15);

                n.bx = centerX + hemisphere * Math.abs(Math.cos(angle)) * rX * 0.88 + (hemisphere * 18 * this.scale);
                n.by = centerY + Math.sin(angle) * rY;
            }
        });
    }

    setViewMode(mode) {
        this.viewMode = mode;
        this.targetMorph = mode === 'layers' ? 1.0 : 0.0;

        const btnBrain = document.getElementById('view-mode-brain');
        const btnLayers = document.getElementById('view-mode-layers');
        if (btnBrain && btnLayers) {
            if (mode === 'brain') {
                btnBrain.classList.add('active-mode-btn');
                btnBrain.classList.remove('text-slate-400');
                btnLayers.classList.remove('active-mode-btn');
                btnLayers.classList.add('text-slate-400');
            } else {
                btnLayers.classList.add('active-mode-btn');
                btnLayers.classList.remove('text-slate-400');
                btnBrain.classList.remove('active-mode-btn');
                btnBrain.classList.add('text-slate-400');
            }
        }
    }

    triggerPulse(intensity = 1.0) {
        this.pulseBurstTimer = 40;
        this.audioStimulation = Math.max(this.audioStimulation, intensity);
        this.neurons.forEach(n => {
            n.energy = 1.0;
        });
    }

    setupInteractions() {
        if (!this.canvas) return;

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = (e.clientX - rect.left) * this.scale;
            this.mouseY = (e.clientY - rect.top) * this.scale;

            let closest = null;
            let minDist = 25 * this.scale;

            this.neurons.forEach(n => {
                const dx = n.x - this.mouseX;
                const dy = n.y - this.mouseY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minDist) {
                    minDist = dist;
                    closest = n;
                }
            });

            this.hoveredNeuron = closest;
            this.updateTooltip(e.clientX, e.clientY);
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.mouseX = -1000;
            this.mouseY = -1000;
            this.hoveredNeuron = null;
            this.hideTooltip();
        });

        this.canvas.addEventListener('click', () => {
            this.triggerPulse(1.0);
        });
    }

    updateTooltip(clientX, clientY) {
        const tooltip = document.getElementById('neural-tooltip');
        if (!tooltip) return;

        if (this.hoveredNeuron) {
            const meta = this.hoveredNeuron.meta;
            document.getElementById('tooltip-title').innerHTML = `
                <span class="w-2 h-2 rounded-full inline-block mr-1.5" style="background: ${meta.color}; box-shadow: 0 0 6px ${meta.color}"></span>
                ${meta.name} <span class="text-[9px] text-slate-400 font-normal">(${meta.type})</span>
            `;
            document.getElementById('tooltip-desc').textContent = meta.desc;
            document.getElementById('tooltip-stats').innerHTML = `
                <span>Shape: <strong class="text-slate-200">${meta.tensor}</strong></span>
                <span>Params: <strong class="text-amber-400">${meta.params}</strong></span>
            `;

            tooltip.classList.remove('hidden');
            const parentRect = this.canvas.getBoundingClientRect();
            const posX = Math.min(clientX - parentRect.left + 15, parentRect.width - 240);
            const posY = Math.min(clientY - parentRect.top + 15, parentRect.height - 110);
            tooltip.style.left = `${posX}px`;
            tooltip.style.top = `${posY}px`;
        } else {
            this.hideTooltip();
        }
    }

    hideTooltip() {
        const tooltip = document.getElementById('neural-tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    }

    update() {
        // Morph interpolation
        this.morphProgress += (this.targetMorph - this.morphProgress) * 0.08;

        // Pulse timer decay
        if (this.pulseBurstTimer > 0) {
            this.pulseBurstTimer--;
        }
        this.audioStimulation *= 0.94;

        // Update Neuron Positions & Energy
        this.neurons.forEach(n => {
            n.x = n.bx + (n.lx - n.bx) * this.morphProgress;
            n.y = n.by + (n.ly - n.by) * this.morphProgress;

            n.pulsePhase += 0.03;
            const idlePulse = Math.sin(n.pulsePhase) * 0.3 + 0.7;
            const boost = n.energy + this.audioStimulation * 1.5;
            n.radius = n.baseRadius * (1 + boost * 0.45);
            n.energy *= 0.95;
        });

        // Update Action Potential Particles
        this.particles.forEach(p => {
            const speedBoost = 1.0 + this.audioStimulation * 2.5 + (this.pulseBurstTimer > 0 ? 2.2 : 0);
            p.progress += p.speed * speedBoost;
            if (p.progress >= 1.0) {
                p.progress = 0.0;
                const connected = this.synapses.filter(s => s.from === p.synapse.to);
                if (connected.length > 0 && Math.random() > 0.3) {
                    p.synapse = connected[Math.floor(Math.random() * connected.length)];
                } else {
                    p.synapse = this.synapses[Math.floor(Math.random() * this.synapses.length)];
                }
            }
        });
    }

    render() {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        ctx.clearRect(0, 0, w, h);

        // 1. Draw Subtle Organic Brain Glow Silhouette (in brain mode)
        if (this.morphProgress < 0.85) {
            const brainAlpha = (1.0 - this.morphProgress) * 0.16;
            const grad = ctx.createRadialGradient(w * 0.5, h * 0.48, 20 * this.scale, w * 0.5, h * 0.48, w * 0.36);
            grad.addColorStop(0, `rgba(99, 102, 241, ${brainAlpha})`);
            grad.addColorStop(0.5, `rgba(6, 182, 212, ${brainAlpha * 0.7})`);
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(w * 0.5, h * 0.48, w * 0.36, 0, Math.PI * 2);
            ctx.fill();
        }

        // 2. Draw Synapses (Axons)
        this.synapses.forEach(syn => {
            const isHoverPath = this.hoveredNeuron && (syn.from === this.hoveredNeuron || syn.to === this.hoveredNeuron);
            const isRecurrent = syn.isRecurrent;

            let alpha = isHoverPath ? 0.95 : 0.08 + this.audioStimulation * 0.22;
            ctx.lineWidth = (isHoverPath ? 2.4 : 0.8) * this.scale;
            ctx.strokeStyle = isHoverPath ? '#38BDF8' : syn.color;
            ctx.globalAlpha = Math.min(1.0, alpha);

            ctx.beginPath();
            if (isRecurrent) {
                const midX = (syn.from.x + syn.to.x) / 2;
                const midY = Math.min(syn.from.y, syn.to.y) - 25 * this.scale;
                ctx.quadraticCurveTo(midX, midY, syn.to.x, syn.to.y);
            } else {
                ctx.moveTo(syn.from.x, syn.from.y);
                ctx.lineTo(syn.to.x, syn.to.y);
            }
            ctx.stroke();
        });

        // 3. Draw Action Potential Traveling Particles
        this.particles.forEach(p => {
            const syn = p.synapse;
            if (!syn || !syn.from || !syn.to) return;

            const t = p.progress;
            let px, py;

            if (syn.isRecurrent) {
                const midX = (syn.from.x + syn.to.x) / 2;
                const midY = Math.min(syn.from.y, syn.to.y) - 25 * this.scale;
                px = (1 - t) * (1 - t) * syn.from.x + 2 * (1 - t) * t * midX + t * t * syn.to.x;
                py = (1 - t) * (1 - t) * syn.from.y + 2 * (1 - t) * t * midY + t * t * syn.to.y;
            } else {
                px = syn.from.x + (syn.to.x - syn.from.x) * t;
                py = syn.from.y + (syn.to.y - syn.from.y) * t;
            }

            ctx.globalAlpha = 0.9;
            ctx.fillStyle = this.audioStimulation > 0.3 ? '#F59E0B' : p.color;
            ctx.shadowColor = p.color;
            ctx.shadowBlur = 8 * this.scale;
            ctx.beginPath();
            ctx.arc(px, py, p.size * (1 + this.audioStimulation * 0.8), 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        });

        // 4. Draw Neurons
        this.neurons.forEach(n => {
            const isHovered = n === this.hoveredNeuron;
            const meta = n.meta;

            ctx.globalAlpha = 1.0;
            ctx.shadowColor = meta.glow;
            ctx.shadowBlur = (isHovered ? 18 : 6 + n.energy * 12) * this.scale;

            if (isHovered || n.energy > 0.4) {
                ctx.strokeStyle = meta.color;
                ctx.lineWidth = 1.5 * this.scale;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius * 1.6, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Neuron Body
            ctx.fillStyle = isHovered ? '#FFFFFF' : meta.color;
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;

            // Tamil Character Label for Output CTC Neurons
            if (n.tamilGrapheme && (this.morphProgress > 0.4 || isHovered)) {
                ctx.fillStyle = isHovered ? '#10B981' : 'rgba(255, 255, 255, 0.75)';
                ctx.font = `${Math.max(9, 11 * this.scale)}px "Mukta Malar", sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(n.tamilGrapheme, n.x + 12 * this.scale, n.y);
            }
        });

        // 5. Draw Layer Labels in Layered Mode
        if (this.morphProgress > 0.55) {
            const padX = this.width * 0.08;
            const colW = (this.width - padX * 2) / (this.layerMeta.length - 1);
            const labelAlpha = (this.morphProgress - 0.55) / 0.45;

            ctx.globalAlpha = labelAlpha;
            this.layerMeta.forEach((meta, idx) => {
                const lx = padX + idx * colW;
                ctx.fillStyle = meta.color;
                ctx.font = `bold ${Math.max(9, 10 * this.scale)}px "JetBrains Mono", monospace`;
                ctx.textAlign = 'center';
                ctx.fillText(meta.type.toUpperCase(), lx, 22 * this.scale);

                ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.font = `${Math.max(8, 9 * this.scale)}px "Plus Jakarta Sans", sans-serif`;
                ctx.fillText(meta.params !== "0" ? `${meta.params} params` : "80 Bins", lx, 34 * this.scale);
            });
            ctx.globalAlpha = 1.0;
        }
    }

    startLoop() {
        const loop = () => {
            this.update();
            this.render();
            this.animationId = requestAnimationFrame(loop);
        };
        loop();
    }
}

// Global visualizer instance
let neuralVisualizerInstance = null;

function initNeuralBrainVisualizer() {
    if (document.getElementById('neural-brain-canvas')) {
        neuralVisualizerInstance = new NeuralBrainVisualizer('neural-brain-canvas');
    }
}

function setNeuralViewMode(mode) {
    if (neuralVisualizerInstance) {
        neuralVisualizerInstance.setViewMode(mode);
    }
}

function triggerNeuralPulse() {
    if (neuralVisualizerInstance) {
        neuralVisualizerInstance.triggerPulse(1.0);
    }
}

function stimulateNeuralBrainWithAudio(intensity) {
    if (neuralVisualizerInstance) {
        neuralVisualizerInstance.triggerPulse(intensity);
    }
}

// Auto init when script loads or DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNeuralBrainVisualizer);
} else {
    initNeuralBrainVisualizer();
}
