import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras
import librosa
import traceback
import audioread.exceptions

model = keras.models.load_model('final_model.h5')
emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad']

def preprocess_audio(audio):
    sr = 22050  # Use the same sampling rate used during recording (you can adjust this accordingly)
    audio = np.array(audio) / np.max(np.abs(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    if mfcc.shape[1] > 100:
        mfcc = mfcc[:, :100]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

def predict_emotion(mfcc):
    predictions = model.predict(mfcc)
    emotion_index = np.argmax(predictions)
    emotion = emotions[emotion_index]
    probability = predictions[0][emotion_index]
    return emotion, probability

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_emotion_endpoint', methods=['POST'])
def predict_emotion_endpoint():
    try:
        if 'archivo_audio' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de audio'}), 400

        archivo = request.files['archivo_audio']
        if archivo.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        audio = np.frombuffer(archivo.read(), dtype=np.int16)
        mfcc = preprocess_audio(audio)

        emotion, probability = predict_emotion(mfcc)

        # Establecemos el URL del GIF según la emoción detectada
        switch_emotion = {
            "Felicidad": "/static/emojis/felizAndres.gif",
            "Ansiedad": "/static/emojis/ansiedadAndres.gif",
            "Curiosidad": "/static/emojis/curiosidadAndres.gif",
            "Tranquilidad": "/static/emojis/tranquilidadAndres.gif"
        }
        gif_url = switch_emotion.get(emotion, "/static/emojis/tranquilidadAndres.gif")

        return jsonify({'emotion': emotion, 'probability': probability, 'gif_url': gif_url}), 200
    except audioread.exceptions.NoBackendError as e:
        return jsonify({'error': 'No se pudo encontrar un backend para leer el archivo de audio.'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Error en el procesamiento del audio.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
