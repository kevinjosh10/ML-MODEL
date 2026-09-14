"""
Interactive in-browser audio recorder for Google Colab using Web Audio API PCM WAV encoding.
Guarantees standard 16-bit PCM WAV recording across Chrome, Firefox, Edge, and Safari.
"""

def get_colab_audio_recorder_js() -> str:
    """Returns JavaScript code that records and encodes audio directly to 16-bit PCM WAV."""
    return """
    const sleep = time => new Promise(resolve => setTimeout(resolve, time));

    var recordAudio = async function(seconds) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      const leftChannelData = [];
      processor.onaudioprocess = function(e) {
        const input = e.inputBuffer.getChannelData(0);
        leftChannelData.push(new Float32Array(input));
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      await sleep(seconds * 1000);
      
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach(track => track.stop());
      
      // Merge audio buffers
      let totalLength = 0;
      for (let i = 0; i < leftChannelData.length; i++) {
        totalLength += leftChannelData[i].length;
      }
      const audioBuffer = new Float32Array(totalLength);
      let offset = 0;
      for (let i = 0; i < leftChannelData.length; i++) {
        audioBuffer.set(leftChannelData[i], offset);
        offset += leftChannelData[i].length;
      }
      
      // Encode to 16-bit PCM WAV
      const wavBuffer = new ArrayBuffer(44 + audioBuffer.length * 2);
      const view = new DataView(wavBuffer);
      
      function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      }
      
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + audioBuffer.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM format
      view.setUint16(22, 1, true); // Mono
      view.setUint32(24, 16000, true); // Sample rate
      view.setUint32(28, 16000 * 2, true); // Byte rate
      view.setUint16(32, 2, true); // Block align
      view.setUint16(34, 16, true); // Bits per sample
      writeString(view, 36, 'data');
      view.setUint32(40, audioBuffer.length * 2, true);
      
      let pcmOffset = 44;
      for (let i = 0; i < audioBuffer.length; i++, pcmOffset += 2) {
        let s = Math.max(-1, Math.min(1, audioBuffer[i]));
        view.setInt16(pcmOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      
      const blob = new Blob([view], { type: 'audio/wav' });
      return new Promise(resolve => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
    };
    """

def record_audio_in_colab(filename: str = "recorded_tamil_voice.wav", duration: int = 3):
    """
    Records audio using browser microphone inside Google Colab and saves to filename.
    Returns the filename if successful, or None if cancelled/failed.
    """
    try:
        from google.colab import output
        from IPython.display import HTML, display, Javascript
        import base64
        import io

        print(f"🎙️ Speak Tamil into your microphone now (Recording for {duration} seconds)...")
        display(Javascript(get_colab_audio_recorder_js()))
        
        js_cmd = f"recordAudio({duration})"
        audio_data = output.eval_js(js_cmd)
        
        if not audio_data or 'base64,' not in audio_data:
            print("⚠️ No audio data received.")
            return None

        header, b64_data = audio_data.split('base64,')
        audio_bytes = base64.b64decode(b64_data)
        
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
            
        print(f"✅ Voice recorded successfully and saved to '{filename}'!")
        return filename
    except Exception as e:
        print(f"ℹ️ Colab recording notice: {e}")
        return None
