import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import librosa

# Definir las emociones a detectar
emotions = ['agradecimiento', 'ansiedad', 'curiosidad', 'expectativa', 'felicidad', 'seguridad', 'tranquilidad']
num_emotions = len(emotions)

# Directorio con los archivos de audio
data_dir = 'C:/Users/Michael/Desktop/IA/Voz/PrIA/Crea_sonido'

# Parámetros del modelo
input_shape = (40, )  # Tamaño de los espectrogramas de mel-frequency cepstral coefficients (MFCCs)
dropout_rate = 0.3
learning_rate = 0.0001
epochs = 50
batch_size = 32

# Función para extraer características de audio utilizando MFCC
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=22050)  # Cargar el archivo de audio
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)  # Calcular los MFCCs
    mfccs = np.mean(mfccs.T, axis=0)  # Promediar los MFCCs a lo largo del tiempo
    return mfccs

# Función para cargar los datos de entrenamiento
def load_data():
    X = []
    y = []
    for i, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_dir):
            file_path = os.path.join(emotion_dir, filename)
            mfccs = extract_features(file_path)
            X.append(mfccs)
            y.append(i)
    X = np.array(X)
    y = to_categorical(y, num_classes=num_emotions)
    return X, y

# Cargar los datos de entrenamiento
X_train, y_train = load_data()

# Construir el modelo de la red neuronal
model = Sequential()
model.add(Dense(256, input_shape=input_shape, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_emotions, activation='softmax'))

# Compilar el modelo
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Definir el punto de control para guardar el modelo con la mejor precisión durante el entrenamiento
checkpoint = ModelCheckpoint('trained_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Entrenar el modelo
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])

# Guardar el modelo entrenado en un archivo .h5
model.save('trained_model.h5')
print("Entrenamiento finalizado. El modelo entrenado ha sido guardado en trained_model.h5.")
