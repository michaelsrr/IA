import os
from flask import Flask, request, jsonify
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow import keras
import librosa

# Cargar el modelo entrenado
model = keras.models.load_model('final_model.h5')

# Definir las emociones disponibles
emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad']

# Función para preprocesar el audio
def preprocess_audio(audio_path):
    # Cargar el archivo de audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Normalizar la amplitud del audio
    audio /= np.max(np.abs(audio))
    
    # Extraer características de audio utilizando Mel-Frequency Cepstral Coefficients (MFCC)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Ajustar la longitud de MFCC a 100 frames
    if mfcc.shape[1] > 100:
        mfcc = mfcc[:, :100]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
    
    # Redimensionar a 3D (1, características, frames, canales)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    
    return mfcc

# Función para predecir la emoción
def predict_emotion(mfcc):
    # Predecir la emoción
    predictions = model.predict(mfcc)
    
    # Obtener la emoción con mayor probabilidad
    emotion_index = np.argmax(predictions)
    emotion = emotions[emotion_index]
    
    # Obtener la probabilidad de la emoción predicha
    probability = predictions[0][emotion_index]
    
    return emotion, probability

# Crear una aplicación Flask
app = Flask(__name__)

# Endpoint para cargar el archivo de audio y realizar la predicción
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion_endpoint():
    try:
        # Verificar si se recibe un archivo de audio
        if 'archivo_audio' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de audio'}), 400

        archivo = request.files['archivo_audio']
        if archivo.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        # Guardar el archivo de audio en un directorio temporal
        audio_path = 'temp_audio.wav'
        archivo.save(audio_path)

        # Preprocesar el audio
        mfcc = preprocess_audio(audio_path)

        # Realizar la predicción de la emoción
        emotion, probability = predict_emotion(mfcc)

        # Eliminar el archivo de audio temporal
        os.remove(audio_path)

        # Devolver los resultados
        return jsonify({'emotion': emotion, 'probability': probability}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
