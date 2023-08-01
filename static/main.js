function processAudio() {
    const audioFileInput = document.getElementById('audioFileInput');
    const resultsDiv = document.getElementById('results');
    const emotionResult = document.getElementById('emotionResult');
    const probabilityResult = document.getElementById('probabilityResult');

    const formData = new FormData();
    formData.append('archivo_audio', audioFileInput.files[0]);

    fetch('/predict_emotion', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        emotionResult.textContent = data.emotion;
        probabilityResult.textContent = data.probability.toFixed(2);
        resultsDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
