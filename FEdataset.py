import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from io import BytesIO


# Mappa delle emozioni
emotion_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}


def load_images_and_labels_from_zip(zip_path, base_dir):
    """
    Carica immagini e etichette da un file ZIP organizzato in directory.
    """
    images = []
    labels = []

    with ZipFile(zip_path, 'r') as zf:
        # Filtra i file basati sulla directory specificata (train, test)
        files = [f for f in zf.namelist() if f.startswith(base_dir) and f.endswith('.jpg')]

        for file in files:
            parts = file.split('/')
            if len(parts) < 3:
                continue  # Ignora file non validi

            emotion = parts[1]  # La seconda parte del path è la categoria
            if emotion not in emotion_map:
                continue

            label = emotion_map[emotion]

            try:
                # Carica e preprocessa l'immagine
                with zf.open(file) as img_file:
                    img = load_img(BytesIO(img_file.read()), target_size=(48, 48), color_mode='grayscale')
                    img_array = img_to_array(img) / 255.0  # Normalizza
                    images.append(img_array)
                    labels.append(label)
            except Exception as e:
                print(f"Errore nel caricamento del file {file}: {e}")

    print(f"Caricate {len(images)} immagini e {len(labels)} etichette dalla directory '{base_dir}'.")
    return np.array(images), np.array(labels)


def split_data(images, labels):
    """
    Divide i dati in set di addestramento e test.
    """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f"Dati divisi: {len(X_train)} training, {len(X_test)} test.")
    return X_train, X_test, y_train, y_test


def prepare_data(zip_path='./datasets/fer2013.zip', base_dir='train'):
    """
    Prepara i dati FER2013 caricandoli direttamente da un file ZIP.
    Restituisce i dati di training e test.
    """
    images, labels = load_images_and_labels_from_zip(zip_path, base_dir)
    return split_data(images, labels)
