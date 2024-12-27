import tensorflow as tf
import numpy as np
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# Definisci le etichette delle emozioni
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_image(image_path):
    """
    Carica e preprocessa l'immagine per il modello .keras.
    """
    # Carica l'immagine e converti in scala di grigi
    img = Image.open(image_path).convert("L")

    # Ridimensiona a 48x48 (dimensione richiesta dal modello)
    img = img.resize((48, 48))

    # Converti l'immagine in array numpy e normalizza i valori
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Aggiungi batch e canale (per il modello Keras)
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Canale (1 per scala di grigi)

    return img_array

def load_keras_model():
    """
    Carica il modello .keras dalla cartella superiore.
    """
    # Percorso del modello nella cartella superiore
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_emotion_model.keras")

    # Verifica che il modello esista
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato al percorso: {model_path}")

    # Carica il modello
    model = tf.keras.models.load_model(model_path)
    return model

def predict_emotion(model, input_image):
    """
    Esegui l'inferenza sul modello .keras e restituisci l'emozione.
    """
    # Esegui la previsione
    predictions = model.predict(input_image)

    # Trova l'etichetta con la probabilità più alta
    predicted_index = np.argmax(predictions[0])
    return EMOTIONS[predicted_index], predictions[0]

if __name__ == "__main__":
    # Carica il modello Keras dalla cartella superiore
    try:
        model = load_keras_model()
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        exit()

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

    # Predici l'emozione
    emotion, probabilities = predict_emotion(model, input_image)

    # Mostra il risultato
    print(f"Immagine selezionata: {image_path}")
    print(f"Emozione rilevata: {emotion}")
    print(f"\nProbabilità", end=" -> ")
    for p, e in zip(probabilities, EMOTIONS):
        print(f"{e}: {p:.4f}", end=" ")
