import sounddevice as sd
import numpy as np
import tensorflow.python.keras as keras

# Cargar el modelo previamente entrenado
model = keras.models.load_model('final_model.h5')

# Definir las emociones
emotions = {
    0: "Agradecimiento",
    1: "Ansiedad",
    2: "Curiosidad",
    3: "Expectativa",
    4: "Felicidad",
    5: "Seguridad",
    6: "Tranquilidad"
}

# Función para realizar la predicción de emociones
def predict_emotion(audio):
    # Realizar la predicción utilizando el modelo cargado
    predictions = model.predict(np.expand_dims(audio, axis=0))
    
    # Obtener el índice de la emoción con mayor probabilidad
    predicted_index = np.argmax(predictions)
    
    # Obtener la emoción correspondiente al índice
    predicted_emotion = emotions[predicted_index]
    
    # Devolver la emoción predicha
    return predicted_emotion

# Función para capturar audio en tiempo real y realizar la identificación de emociones
def capture_audio():
    # Configurar los parámetros de grabación
    sample_rate = 44100  # Frecuencia de muestreo en Hz
    duration = 5  # Duración de la grabación en segundos

    # Capturar audio desde el micrófono
    print("Habla ahora...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    # Realizar la identificación de emociones
    predicted_emotion = predict_emotion(audio)

    # Mostrar la emoción identificada
    print("Emoción identificada:", predicted_emotion)

# Capturar audio y realizar la identificación de emociones
capture_audio()
