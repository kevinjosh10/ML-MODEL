"""
Interactive in-browser audio recorder for Google Colab using JavaScript AudioContext.
"""

def get_colab_audio_recorder_js() -> str:
    """Returns JavaScript code to record audio directly from Google Colab's notebook interface."""
    return """
    const sleep = time => new Promise(resolve => setTimeout(resolve, time));
    const b2text = blob => new Promise(resolve => {
      const reader = new FileReader();
      reader.onloadend = e => resolve(e.srcElement.result);
      reader.readAsDataURL(blob);
    });

    var record = async function(sec) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
      chunks = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      recorder.start();
      await sleep(sec * 1000);
      recorder.stop();
      await sleep(200);
      blob = new Blob(chunks, { type: 'audio/wav' });
      return await b2text(blob);
    }
    """

def record_audio_in_colab(filename: str = "recorded_tamil_voice.wav", duration: int = 3):
    """
    Records audio using browser microphone inside Google Colab and saves to filename.
    """
    try:
        from google.colab import output
        from IPython.display import HTML, display, Javascript
        import base64
        import io
        import soundfile as sf
        import librosa

        print(f"🎙️ Speak Tamil into your microphone for {duration} seconds...")
        display(Javascript(get_colab_audio_recorder_js()))
        
        js_cmd = f"record({duration})"
        audio_data = output.eval_js(js_cmd)
        
        header, b64_data = audio_data.split('base64,')
        audio_bytes = base64.b64decode(b64_data)
        
        # Load audio bytes and resample/save
        audio_io = io.BytesIO(audio_bytes)
        data, sr = sf.read(audio_io)
        
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        sf.write(filename, data, sr)
        print(f"✅ Recording saved successfully as '{filename}'!")
        return filename
    except Exception as e:
        print(f"Colab recording note: {e}")
        return None
