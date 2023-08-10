let mediaRecorder;
let recordedChunks = [];
let isRecording = false;

function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Tu navegador no soporta la API de Web Audio para grabación de audio.');
        return;
    }

    const constraints = { audio: true };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = function() {
                const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                processAudio(audioUrl);
            };

            mediaRecorder.start();
            isRecording = true;
            toggleRecordingButton();
        })
        .catch(function(err) {
            console.error('Error al acceder al dispositivo de audio:', err);
        });
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        toggleRecordingButton();
    }
}

function toggleRecordingButton() {
    const recordButton = document.getElementById('recordButton');
    if (isRecording) {
        recordButton.textContent = 'Detener Grabación';
        recordButton.classList.add('recording');
    } else {
        recordButton.textContent = 'Iniciar Grabación';
        recordButton.classList.remove('recording');
    }
}

function processAudio(audioUrl) {
    const formData = new FormData();
    formData.append('archivo_audio', new Blob([recordedChunks[0]], { type: 'audio/wav' }));

    fetch('/predict_emotion_endpoint', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        showResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function showResults(data) {
    const resultsDiv = document.getElementById('results');
    const emotionResult = document.getElementById('emotionResult');
    const probabilityResult = document.getElementById('probabilityResult');
    const gifImg = document.getElementById('gifImg'); 

    if (data.emotion && data.probability) {
        emotionResult.textContent = data.emotion;
        probabilityResult.textContent = data.probability.toFixed(2);
        resultsDiv.style.display = 'block';

        gifImg.src = data.gif_url; 

        const modal = document.getElementById('myModal');
        modal.style.display = "block";
    } else {
        emotionResult.textContent = 'Error en la predicción.';
        probabilityResult.textContent = '';
        resultsDiv.style.display = 'block';
    }
}

function showResults(data) {
    const resultsDiv = document.getElementById('results');
    const emotionResult = document.getElementById('emotionResult');
    const probabilityResult = document.getElementById('probabilityResult');

    if (data.emotion && data.probability) {
        emotionResult.textContent = data.emotion;
        probabilityResult.textContent = data.probability.toFixed(2); // Fixed to 2 decimal places
        resultsDiv.style.display = 'block';

        // Show the GIF based on the detected emotion
        const gifImg = document.getElementById('gifImg');
        gifImg.src = data.gif_url;
    } else {
        // Mostrar mensaje de error en lugar de resultados incompletos.
        emotionResult.textContent = 'Error en la predicción.';
        probabilityResult.textContent = '';
        resultsDiv.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    recordButton.addEventListener('click', function() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
});
     

document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    recordButton.addEventListener('click', function() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    const modal = document.getElementById('myModal');
    const closeButton = document.getElementsByClassName("close")[0];

    closeButton.onclick = function() {
        modal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    }
});
