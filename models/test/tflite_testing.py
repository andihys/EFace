import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Definisci le etichette delle emozioni
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_image(image_path):
    """
    Carica e preprocessa l'immagine per il modello TFLite con input INT8.
    """
    # Carica l'immagine e converti in scala di grigi
    img = Image.open(image_path).convert("L")

    # Ridimensiona a 48x48 (dimensione richiesta dal modello)
    img = img.resize((48, 48))

    # Converti l'immagine in array numpy
    img_array = np.array(img, dtype=np.float32)

    # Normalizza i valori nell'intervallo [0, 255]
    img_array = img_array / 255.0 * 255.0  # Proporziona i valori per INT8

    # Aggiungi batch e canale
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Canale (1 per scala di grigi)

    # Converte in INT8
    img_array = np.rint(img_array).astype(np.int8)  # Usa np.rint per valori coerenti

    return img_array


def load_tflite_model(modelname):
    """
    Carica il modello TFLite.
    """
    # Percorso del modello nella cartella superiore
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), modelname)

    # Verifica che il modello esista
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato al percorso: {model_path}")
    # Carica l'interprete TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_emotion(interpreter, input_image):
    """
    Esegui l'inferenza sul modello TFLite e restituisci l'emozione.
    """
    # Ottieni i dettagli di input e output del modello
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Imposta il tensore di input
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Esegui l'inferenza
    interpreter.invoke()

    # Ottieni il risultato
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Trova l'etichetta con la probabilità più alta
    predicted_index = np.argmax(predictions)
    return EMOTIONS[predicted_index], predictions

if __name__ == "__main__":
    # Percorso del modello TFLite
    tflite_model = "emotion_model.tflite"

    # Usa Tkinter per aprire una finestra di selezione file
    Tk().withdraw()  # Nascondi la finestra principale di Tkinter
    print("Seleziona un'immagine...")
    image_path = askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    # Verifica che un'immagine sia stata selezionata
    if not image_path:
        print("Nessuna immagine selezionata.")
        exit()

    # Carica e preprocessa l'immagine
    input_image = load_image(image_path)

    # Carica il modello TFLite
    interpreter = load_tflite_model(tflite_model)

    # Predici l'emozione
    emotion, probabilities = predict_emotion(interpreter, input_image)

    # Mostra il risultato
    print(f"Immagine selezionata: {image_path}")
    print(f"Emozione rilevata: {emotion}")
    print(f"Probabilità", end=" -> ")
    for p, e in zip(probabilities, EMOTIONS):
        print(f"{e}: {p:.4f}", end=" ")
