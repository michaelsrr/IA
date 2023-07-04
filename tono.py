# Importar las librerías necesarias
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Definir la duración de la grabación en segundos
duration = 5

# Definir la frecuencia de muestreo en Hz
fs = 44100

# Grabar el sonido del micrófono
print("Habla por el micrófono durante", duration, "segundos")
sound = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Grabación terminada")

# Guardar el sonido como un archivo wav
wav.write("sound.wav", fs, sound)

# Leer el archivo wav y obtener la señal y la frecuencia de muestreo
rate, signal = wav.read("sound.wav")

# Calcular el espectro de frecuencias de la señal usando la transformada de Fourier
freqs = np.fft.rfftfreq(len(signal), d=1/rate)
amps = np.abs(np.fft.rfft(signal))

# Encontrar la frecuencia dominante, que corresponde al tono de voz
peak_freq = freqs[np.argmax(amps)]

# Clasificar el tono de voz según la frecuencia dominante
if peak_freq < 165:
    tone = "bajo"
elif peak_freq < 255:
    tone = "medio"
else:
    tone = "alto"

# Mostrar el resultado
print("El tono de voz es:", tone)


