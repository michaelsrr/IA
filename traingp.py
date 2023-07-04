import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Definir las emociones y sus etiquetas correspondientes
emotions = {
    'Tranquilidad': 0,
    'Seguridad': 1,
    'Expectativa': 2,
    'Felicidad': 3,
    'Curiosidad': 4
}

# Directorio que contiene los archivos de audio para cada emoción
data_dir = 'C:/Users/Michael/Desktop/IA/Voz/PrIA/Crea_sonido'

# Lista para almacenar los datos de audio y las etiquetas
audio_data = []
labels = []

# Recorrer las emociones y cargar los archivos de audio correspondientes
for emotion, label in emotions.items():
    emotion_dir = os.path.join(data_dir, emotion)
    audio_files = os.listdir(emotion_dir)
    
    # Recorrer los archivos de audio y cargar los datos
    for audio_file in audio_files:
        audio_path = os.path.join(emotion_dir, audio_file)
        
        # Cargar el archivo de audio utilizando librosa
        audio, sr = librosa.load(audio_path, sr=None)  # sr=None para mantener la frecuencia de muestreo original
        
        # Realizar cualquier preprocesamiento adicional que necesites en los datos de audio
        
        # Agregar los datos de audio y etiquetas a las listas
        audio_data.append(audio)
        labels.append(label)

# Convertir las listas en matrices numpy
audio_data = np.array(audio_data)
labels = np.array(labels)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

# Preprocesamiento de los datos
# Asegurarse de que los datos de audio tengan las dimensiones adecuadas para la entrada a la CNN
max_length = max(len(audio) for audio in audio_data)
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_length, dtype='float32')
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_length, dtype='float32')

# Convertir las etiquetas en codificación one-hot
num_classes = len(emotions)
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(max_length, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Añadir una dimensión extra para el canal (channel) en los datos de entrenamiento y prueba
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

# Entrenar el modelo
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Evaluar el modelo
loss, accuracy = model.evaluate(test_data, test_labels)
print("Loss:", loss)
print("Accuracy:", accuracy)




