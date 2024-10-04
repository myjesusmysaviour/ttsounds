from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import os

app = Flask(__name__)
CORS(app)

# Load models and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector containing speaker's voice characteristics
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/tts', methods=['POST'])
def text_to_speech():
    text = request.json['text']
    
    inputs = processor(text=text, return_tensors="pt")
    
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save the audio to a file
    output_file = "output.wav"
    sf.write(output_file, speech.numpy(), samplerate=16000)
    
    return send_file(output_file, mimetype="audio/wav")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))