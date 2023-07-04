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

# Función para capturar el audio en tiempo real y realizar la predicción
def capture_and_predict():
    # Configurar la frecuencia de muestreo y la duración de la captura
    fs = 22050
    duration = 5  # Duración de la captura en segundos
    
    # Capturar el audio
    print('Habla para identificar la emoción...')
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()
    
    # Guardar el audio en un archivo temporal
    audio_path = 'temp_audio.wav'
    sf.write(audio_path, audio, fs)
    
    # Preprocesar el audio
    mfcc = preprocess_audio(audio_path)
    
    # Realizar la predicción de la emoción
    emotion, probability = predict_emotion(mfcc)
    
    # Mostrar los resultados
    print(f'Emoción identificada: {emotion}')
    print(f'Probabilidad: {probability}')

# Ejecutar la función de captura y predicción
capture_and_predict()
