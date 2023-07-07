<?php
// Importar las bibliotecas necesarias
require 'vendor/autoload.php';
use TensorFlow\TensorFlow;
use TensorFlow\Keras\Model;
use Misd\SoundTouch\SoundTouch;
use Phpml\Audio\Feature\Extractor\Mfcc;

// Cargar el modelo entrenado
$model = new Model;
$model->load('final_model.h5');

// Definir las emociones disponibles
$emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad'];

// Función para preprocesar el audio
function preprocess_audio($audio_data) {
    // Convertir el audio a un archivo temporal
    $audio_path = 'temp_audio.wav';
    file_put_contents($audio_path, $audio_data);
    
    // Cargar el archivo de audio
    $audio = file_get_contents($audio_path);
    $waveData = unpack("S*", $audio);
    
    // Normalizar la amplitud del audio
    $maxValue = max($waveData);
    foreach ($waveData as &$value) {
        $value /= $maxValue;
    }
    
    // Extraer características de audio utilizando Mel-Frequency Cepstral Coefficients (MFCC)
    $mfcc = Mfcc::extract($waveData);
    
    // Ajustar la longitud de MFCC a 100 frames
    if (count($mfcc[0]) > 100) {
        $mfcc = array_slice($mfcc, 0, 100);
    } else {
        $mfcc = array_pad($mfcc, 100, array_fill(0, count($mfcc[0]), 0));
    }
    
    // Redimensionar a 3D (1, características, frames, canales)
    $mfcc = [[[[$mfcc]]]];
    
    return $mfcc;
}

// Función para predecir la emoción
function predict_emotion($mfcc) {
    // Predecir la emoción
    $predictions = $model->predict($mfcc);
    
    // Obtener la emoción con mayor probabilidad
    $emotion_index = array_keys($predictions, max($predictions))[0];
    $emotion = $emotions[$emotion_index];
    
    // Obtener la probabilidad de la emoción predicha
    $probability = $predictions[0][$emotion_index];
    
    return [$emotion, $probability];
}

// Verificar si se recibió un archivo de audio
if ($_FILES['audio']['error'] === UPLOAD_ERR_OK) {
    // Obtener el archivo de audio
    $audioData = file_get_contents($_FILES['audio']['tmp_name']);

    // Preprocesar el audio
    $mfcc = preprocess_audio($audioData);

    // Realizar la predicción de la emoción
    [$emotion, $probability] = predict_emotion($mfcc);

    // Crear un array con los resultados
    $result = [
        'emotion' => $emotion,
        'probability' => $probability
    ];

    // Devolver los resultados en formato JSON
    header('Content-Type: application/json');
    echo json_encode($result);
}
?>
