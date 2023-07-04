// Variables globales
var model;
var emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad'];
var audioContext;
var stream;
var recorder;

// Función para cargar el modelo
async function loadModel() {
  model = await tf.loadLayersModel('model.json');
  console.log('Modelo cargado');
}

// Función para preprocesar el audio
function preprocessAudio(audioData) {
  const maxAmplitude = Math.max(...audioData.map(Math.abs));
  const normalizedAudio = audioData.map(sample => sample / maxAmplitude);

  const mfcc = librosa.mfcc(normalizedAudio);
  const paddedMfcc = tf.pad(mfcc, [[0, 0], [0, Math.max(0, 100 - mfcc.shape[1])]]);

  const reshapedMfcc = paddedMfcc.reshape([1, mfcc.shape[0], 100, 1]);
  return reshapedMfcc;
}

// Función para realizar la predicción de la emoción
async function predictEmotion(mfcc) {
  const predictions = await model.predict(mfcc).data();
  const emotionIndex = predictions.indexOf(Math.max(...predictions));
  const emotion = emotions[emotionIndex];
  const probability = predictions[emotionIndex];
  
  return { emotion, probability };
}

// Función para iniciar la grabación de audio
function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(function (streamData) {
      audioContext = new AudioContext();
      stream = streamData;
      recorder = new MediaRecorder(streamData);

      const audioChunks = [];

      recorder.addEventListener('dataavailable', function (event) {
        audioChunks.push(event.data);
      });

      recorder.addEventListener('stop', function () {
        const audioBlob = new Blob(audioChunks);
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audioContext.decodeAudioData(audioBlob, function (buffer) {
          const audioData = Array.from(buffer.getChannelData(0));
          const mfcc = preprocessAudio(audioData);
          predictEmotion(mfcc).then(function (result) {
            console.log('Emoción identificada:', result.emotion);
            console.log('Probabilidad:', result.probability);
          });
        });
      });

      recorder.start();
      console.log('Grabando audio...');
    })
    .catch(function (error) {
      console.error('Error al acceder al micrófono:', error);
    });
}

// Función para detener la grabación de audio
function stopRecording() {
  if (recorder) {
    recorder.stop();
    stream.getTracks().forEach(function (track) {
      track.stop();
    });
    console.log('Grabación finalizada');
  }
}

// Cargar el modelo al cargar la página
window.onload = function () {
  loadModel();
};
