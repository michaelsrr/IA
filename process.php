<?php

require_once 'libraries/path/to/keras.php';
require_once 'libraries/path/to/librosa.php';
require_once 'libraries/path/to/numpy.php';
require_once 'libraries/path/to/sounddevice.php';
require_once 'libraries/path/to/soundfile.php';


header('Content-Type: application/json');

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    if (isset($_FILES["audio"]) && $_FILES["audio"]["error"] === UPLOAD_ERR_OK) {
        $uploadDir = "C:/xampp/htdocs/reconocimiento-emociones/";
        $uploadFile = $uploadDir . basename($_FILES["audio"]["name"]);

        if (move_uploaded_file($_FILES["audio"]["tmp_name"], $uploadFile)) {
            // Cargar el modelo entrenado
            require_once 'path/to/keras.php';
            $model = keras\models\load_model('final_model.h5');

            // Definir las emociones disponibles
            $emotions = ['Agradecimiento', 'Ansiedad', 'Curiosidad', 'Expectativa', 'Felicidad', 'Seguridad', 'Tranquilidad'];

            // Función para preprocesar el audio
            function preprocess_audio($audio_path) {
                require_once 'path/to/librosa.php';
                require_once 'path/to/numpy.php';
                
                // Cargar el archivo de audio
                $audio = librosa\load($audio_path, sr=NULL);
                
                // Normalizar la amplitud del audio
                $audio /= np\max(np\abs($audio));
                
                // Extraer características de audio utilizando Mel-Frequency Cepstral Coefficients (MFCC)
                $mfcc = librosa\feature\mfcc($audio, sr=NULL, n_mfcc=13);
                
                // Ajustar la longitud de MFCC a 100 frames
                if ($mfcc.shape[1] > 100) {
                    $mfcc = $mfcc[:, :100];
                } else {
                    $mfcc = np\pad($mfcc, array(array(0, 0), array(0, 100 - $mfcc.shape[1])) , mode='constant');
                }
                
                // Redimensionar a 3D (1, características, frames, canales)
                $mfcc = np\expand_dims($mfcc, axis=0);
                $mfcc = np\expand_dims($mfcc, axis=-1);
                
                return $mfcc;
            }

            // Función para predecir la emoción
            function predict_emotion($mfcc) {
                // Predecir la emoción
                $predictions = $model->predict($mfcc);
                
                // Obtener la emoción con mayor probabilidad
                $emotion_index = np\argmax($predictions);
                $emotion = $emotions[$emotion_index];
                
                // Obtener la probabilidad de la emoción predicha
                $probability = $predictions[0][$emotion_index];
                
                return array("emotion" => $emotion, "probability" => $probability);
            }

            // Función para capturar el audio en tiempo real y realizar la predicción
            function capture_and_predict() {
                require_once 'path/to/sounddevice.php';
                require_once 'path/to/soundfile.php';
                
                // Configurar la frecuencia de muestreo y la duración de la captura
                $fs = 22050;
                $duration = 5;  // Duración de la captura en segundos
                
                // Capturar el audio
                $audio = sd\rec(int($fs * $duration), array("samplerate"=>$fs, "channels"=>1));
                sd\wait();
                
                // Guardar el audio en un archivo temporal
                $audio_path = 'temp_audio.wav';
                sf\write($audio_path, $audio, $fs);
                
                // Preprocesar el audio
                $mfcc = preprocess_audio($audio_path);
                
                // Realizar la predicción de la emoción
                $result = predict_emotion($mfcc);
                
                // Devolver los resultados como una respuesta JSON
                echo json_encode($result);
            }

            // Ejecutar la función de captura y predicción
            capture_and_predict();
        } else {
            echo json_encode(array("error" => "Error al mover el archivo."));
        }
    } else {
        echo json_encode(array("error" => "No se envió ningún archivo de audio o ocurrió un error al subirlo."));
    }
} else {
    echo json_encode(array("error" => "Acceso no válido."));
}
?>
