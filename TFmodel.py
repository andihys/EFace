import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from FEdataset import prepare_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disattiva warning non critici


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
        './models/best_emotion_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
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
    convert_to_tflite('best_emotion_model.keras', 'emotion_model.tflite')


def convert_to_tflite(model_path, tflite_path, quantize=True):
    """
    Converte un modello TensorFlow in formato TensorFlow Lite con opzione per la quantizzazione
    così da includere ottimizzazioni che riducono la dimensione del modello e
    ne migliorano le prestazioni su dispositivi mobili.

    La quantizzazione consiste nel ridurre la precisione dei numeri utilizzati nei calcoli
    (ad esempio, passare da 32-bit float a 8-bit integer) per rendere il modello più leggero e veloce,
    senza sacrificare troppo l'accuratezza.

    Args:
        model_path (str): Path al modello TensorFlow (.keras).
        tflite_path (str): Path per salvare il modello TFLite (.tflite).
        quantize (bool): Se True, applica la quantizzazione.

    """
    # Carica il modello TensorFlow
    model = tf.keras.models.load_model(model_path)

    # Configura il convertitore TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # Abilita la quantizzazione full-integer
        print("Eseguendo la quantizzazione del modello...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Imposta un rappresentante dei dati per la quantizzazione full-integer
        def representative_dataset_gen():
            X_train, _, _, _ = prepare_data()
            for i in range(min(100, len(X_train))):  # Usa fino a 100 campioni rappresentativi
                yield [X_train[i].reshape(1, 48, 48, 1).astype("float32")]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # Input quantizzato
        converter.inference_output_type = tf.int8  # Output quantizzato

    # Converte il modello
    tflite_model = converter.convert()

    # Salva il modello TFLite
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
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

    # Path del modello quantizzato
    tflite_path = 'emotion_model.tflite'

    # Verifica del modello TFLite
    verify_tflite_model(tflite_path)

