import os
import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint

# Definir las emociones
emotions = {
    'agradecimiento': 0,
    'ansiedad': 1,
    'curiosidad': 2,
    'expectativa': 3,
    'felicidad': 4,
    'seguridad': 5,
    'tranquilidad': 6
}

# Directorio de los audios
audio_dir = 'C:/Users/Michael/Desktop/IA/Voz/PrIA/Crea_sonido'

# Obtener las rutas de los archivos de audio y las etiquetas correspondientes
audio_paths = []
labels = []
for emotion, label in emotions.items():
    emotion_dir = os.path.join(audio_dir, emotion)
    audio_files = os.listdir(emotion_dir)
    for audio_file in audio_files:
        audio_path = os.path.join(emotion_dir, audio_file)
        audio_paths.append(audio_path)
        labels.append(label)

# Convertir las etiquetas a números enteros
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Definir el tamaño deseado para las características MFCC
max_len = 100

# Función para extraer características de los audios utilizando librosa
def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)  # Cargar el archivo de audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extraer coeficientes MFCC
    # Ajustar el tamaño de las características
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    else:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# Extraer características de los audios y almacenarlos en una matriz
features = []
for audio_path in audio_paths:
    mfccs = extract_features(audio_path)
    features.append(mfccs)
features = np.array(features)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Expandir dimensiones para que se ajusten a la entrada de la red neuronal
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convertir las etiquetas a codificación one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Construir el modelo de la red neuronal
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Guardar los pesos del modelo con mejor precisión durante el entrenamiento
checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Guardar el modelo entrenado
model.save('final_model.h5')
