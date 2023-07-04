# Importar las librerías necesarias
import speech_recognition as sr
import librosa
import numpy as np

# Crear un objeto de reconocimiento de voz
r = sr.Recognizer()

# Definir una función para calcular el ritmo de voz en sílabas por segundo
def calcular_ritmo(archivo):
  # Cargar el archivo de audio y extraer la señal y la frecuencia de muestreo
  senal, frecuencia = librosa.load(archivo)
  # Calcular la duración del audio en segundos
  duracion = len(senal) / frecuencia
  # Usar librosa para segmentar la señal en sílabas
  segmentos = librosa.effects.split(senal)
  # Contar el número de sílabas
  num_silabas = len(segmentos)
  # Calcular el ritmo de voz en sílabas por segundo
  ritmo = num_silabas / duracion
  return ritmo

# Definir una función para clasificar el ritmo de voz en rápido, medio o pausado
def clasificar_ritmo(ritmo):
  # Definir unos umbrales arbitrarios para cada categoría
  umbral_rapido = 7
  umbral_medio = 4
  # Comparar el ritmo con los umbrales y devolver la categoría correspondiente
  if ritmo > umbral_rapido:
    return "rápido"
  elif ritmo > umbral_medio:
    return "medio"
  else:
    return "pausado"

# Definir una función para reconocer el ritmo de voz de una persona que habla por micrófono del computador
def reconocer_ritmo():
  # Usar el micrófono como fuente de audio
  with sr.Microphone() as source:
    # Ajustar el nivel de ruido ambiental
    r.adjust_for_ambient_noise(source)
    # Indicar al usuario que empiece a hablar
    print("Habla ahora")
    # Grabar el audio del usuario
    audio = r.listen(source)
    # Intentar transcribir el audio a texto usando Google Speech Recognition
    try:
      texto = r.recognize_google(audio, language="es-ES")
      print("Has dicho: " + texto)
      # Guardar el audio en un archivo temporal
      archivo = "temp.wav"
      with open(archivo, "wb") as f:
        f.write(audio.get_wav_data())
      # Calcular el ritmo de voz del archivo
      ritmo = calcular_ritmo(archivo)
      print("Tu ritmo de voz es: {:.2f} sílabas por segundo".format(ritmo))
      # Clasificar el ritmo de voz en rápido, medio o pausado
      categoria = clasificar_ritmo(ritmo)
      print("Tu categoría de ritmo de voz es: " + categoria)
    except sr.UnknownValueError:
      print("No se ha podido reconocer el audio")
    except sr.RequestError as e:
      print("Ha ocurrido un error al conectarse con Google Speech Recognition; {0}".format(e))

# Llamar a la función principal
reconocer_ritmo()



