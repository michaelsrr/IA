<?php
// Verificar si se recibió un archivo de audio
if (isset($_FILES['audio'])) {
  $audioFile = $_FILES['audio']['tmp_name'];

  // Realizar el procesamiento de la predicción aquí

  // Ejemplo de respuesta JSON
  $response = [
    'emotion' => 'Felicidad',
    'probability' => 0.85
  ];

  // Devolver la respuesta como JSON
  header('Content-Type: application/json');
  echo json_encode($response);
} else {
  // Devolver un error si no se recibió el archivo de audio
  http_response_code(400);
  echo 'Error: No se recibió el archivo de audio.';
}
?>
