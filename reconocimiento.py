import librosa
import numpy as np
import tensorflow as tf
import pyaudio
import sounddevice as sd
import speech_recognition as sr

# Cargar el modelo entrenado
model = tf.keras.models.load_model('trained_model.h5')

# Definir las emociones disponibles
emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad']

# Función para preprocesar el audio
def preprocess_audio(audio, sample_rate):
    # Extraer las características del audio utilizando MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Transponer las características
    mfccs = np.transpose(mfccs)

    return mfccs

# Función para predecir la emoción en base al audio
def predict_emotion(audio, sample_rate):
    # Preprocesar el audio
    mfccs = preprocess_audio(audio, sample_rate)

    # Rellenar o recortar las características para tener una longitud fija de 40
    if mfccs.shape[0] > 40:
        mfccs = mfccs[:40, :]
    else:
        mfccs = np.pad(mfccs, ((0, 40 - mfccs.shape[0]), (0, 0)), mode='constant')

    # Expandir las dimensiones del tensor para que coincida con la forma esperada por el modelo
    mfccs = np.expand_dims(mfccs, axis=0)

    # Realizar la predicción
    predictions = model.predict(mfccs)
    predicted_emotion_index = np.argmax(predictions)
    
    if predicted_emotion_index < len(emotions):
        predicted_emotion = emotions[predicted_emotion_index]
    else:
        predicted_emotion = "Emoción desconocida"
    
    return predicted_emotion

# Configuración de la grabación de audio
fs = 22050  # Frecuencia de muestreo
duration = 5  # Duración de la grabación en segundos

# Función para capturar la entrada de voz desde el micrófono
def record_audio(sample_rate):
    print("Habla ahora...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Grabación finalizada.")
    return audio.flatten()

# Función para reconocer el texto en la grabación de voz en español
def recognize_speech(audio, sample_rate):
    r = sr.Recognizer()
    audio_data = sr.AudioData(audio.tobytes(), sample_rate=sample_rate, sample_width=audio.dtype.itemsize)
    text = ""

    return text

# Capturar la entrada de voz desde el micrófono, reconocer el texto y realizar la identificación de emociones
def identify_emotion():
    while True:
        audio = record_audio(fs)
        text = recognize_speech(audio, fs)
        predicted_emotion = predict_emotion(audio, fs)
        print('Texto grabado:', text)
        print('La emoción detectada es:', predicted_emotion)

# Iniciar la identificación de emociones
identify_emotion()
