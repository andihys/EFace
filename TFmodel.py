import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from FEdataset import prepare_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disattiva warning non critici
keras_path = 'models/best_emotion_model.keras'
tflite_path = 'models/emotion_model.tflite'

def create_model(input_shape, num_classes):
    """
    Crea un modello CNN per il riconoscimento delle emozioni.
    Include tecniche di miglioramento delle prestazioni.
    """
    model = models.Sequential([
        # Aggiungi un livello Input esplicito
        Input(shape=input_shape),

        # Primo blocco di convoluzione
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Secondo blocco di convoluzione
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Terzo blocco di convoluzione
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Livelli completamente connessi
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_save_model():
    """
    Addestra il modello e lo salva in formato TFLite con migliori pratiche.
    """
    # Carica i dati pre-elaborati
    X_train, X_test, y_train, y_test = prepare_data()

    # Reshape dei dati per TensorFlow
    X_train = X_train.reshape((-1, 48, 48, 1))  # 48x48 immagini in scala di grigi
    X_test = X_test.reshape((-1, 48, 48, 1))

    # Converte le etichette in formato one-hot
    num_classes = len(set(y_train))  # Numero di classi (es: 7 emozioni)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Crea il modello
    model = create_model(input_shape=(48, 48, 1), num_classes=num_classes)

    # Compila il modello con un ottimizzatore e una funzione di perdita avanzati
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks per migliorare le prestazioni
    checkpoint_cb = callbacks.ModelCheckpoint(
        keras_path, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping_cb = callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor='val_accuracy')

    # Addestra il modello con dati aumentati
    print("Inizio dell'addestramento con data augmentation...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_gen = datagen.flow(X_train, y_train, batch_size=64)

    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # Valutazione finale
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Accuratezza finale sul test set: {test_accuracy * 100:.2f}%")

    # Converte e salva in formato TensorFlow Lite
    convert_to_tflite_optimized(keras_path, tflite_path)


def convert_to_tflite_optimized(model_path, tflite_path, quantization="int8"):
    """
    Converte un modello TensorFlow in formato TensorFlow Lite con opzioni di quantizzazione avanzate.

    Args:
        model_path (str): Path al modello TensorFlow (.keras).
        tflite_path (str): Path per salvare il modello TFLite (.tflite).
        quantization (str): Tipo di quantizzazione:
            - "int8": Quantizzazione full-integer.
            - "float16": Quantizzazione a precisione ridotta (float16).
            - "none": Nessuna quantizzazione.
    """
    # Carica il modello TensorFlow
    model = tf.keras.models.load_model(model_path)

    # Configura il convertitore TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Applica la quantizzazione selezionata
    if quantization == "int8":
        print("Eseguendo la quantizzazione full-integer...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Dataset rappresentativo per la quantizzazione full-integer
        def representative_dataset_gen():
            X_train, _, _, _ = prepare_data()  # Assumi che prepare_data restituisca i dati preprocessati
            for i in range(min(100, len(X_train))):  # Usa fino a 100 campioni rappresentativi
                yield [X_train[i].reshape(1, 48, 48, 1).astype("float32")]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # Input quantizzato
        converter.inference_output_type = tf.int8  # Output quantizzato

    elif quantization == "float16":
        print("Eseguendo la quantizzazione float16...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    else:
        print("Nessuna quantizzazione applicata.")

    # Converte il modello
    tflite_model = converter.convert()

    # Salva il modello TFLite
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Confronta la dimensione dei modelli
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(tflite_path)
    print(f"Modello originale: {original_size / 1024:.2f} KB")
    print(f"Modello TFLite: {tflite_size / 1024:.2f} KB")
    print(f"Riduzione dimensione: {100 * (1 - tflite_size / original_size):.2f}%")
    print(f"Modello convertito e salvato come '{tflite_path}'.")



def verify_tflite_model(tflite_path):
    """
    Verifica il modello TFLite quantizzato:
    - Tipo di input/output
    - Dimensioni del file
    """
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n--- Verifica del modello TFLite ---")
    print(f"Tipo di input: {input_details[0]['dtype']}")
    print(f"Tipo di output: {output_details[0]['dtype']}")
    print(f"Dimensioni del modello (in byte): {len(tflite_model)}")
    print("--- Fine verifica ---\n")


if __name__ == "__main__":
    # Addestramento e salvataggio del modello
    train_and_save_model()

    # modello TFLite
    convert_to_tflite_optimized("best_emotion_model.keras", "model_optimized.tflite", quantization="float16")

