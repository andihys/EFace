package com.example.emotionalfaces;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;


public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;
    private ImageView imageView;
    private TextView textView;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private Bitmap capturedBitmap;
    private Interpreter tflite;
    private NnApiDelegate nnApiDelegate;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 10;
    private static final int TIME_DETECTION = 2000;
    private static final int WAKE_UP_TIME = 1000;
    private static final String MODEL_PATH = "emotion_model.tflite";
    private static final int IMAGE_SIZE = 48; // Dimensione richiesta dal modello
    private static final String[] EMOTIONS = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"};
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Trova i componenti nel layout
        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
        Button buttonTakePicture = findViewById(R.id.buttonTakePicture);
        Button buttonProcess = findViewById(R.id.buttonProcess);

        // Configura il thread per CameraX
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Richiedi permessi se necessario
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }

        // Carica il modello TFLite
        try {
            tflite = loadModel(MODEL_PATH);
        } catch (IOException e) {
            Log.e("MainActivity", "ML model TFLite upload error: " + e.getMessage());
        }
        // azione per il pulsante "Take Picture"
        buttonTakePicture.setOnClickListener(v -> takePhoto());

        // azione per processare l'immagine
        buttonProcess.setOnClickListener(v -> {
            inference();
        });

        startBackgroundprocess();
    }


    private void inference(){
        if (capturedBitmap != null) {
            String emotion = processImage_8b(capturedBitmap);
            //String emotion = processImage_16f(capturedBitmap);
            textView.setText(getString(R.string.emotion_detected) + " " + emotion);
        } else {
            // Toast.makeText(this, "Take a picture!", Toast.LENGTH_SHORT).show();
            Log.e("MainActivity", "No image to process");
        }
    }


    private void startBackgroundprocess() {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        executorService.execute(() -> {
            try {
                Thread.sleep(WAKE_UP_TIME);
            } catch (Exception e) {
                Log.e("Wake up error", "Error on wake up sleeping", e);
            }
            while (true) {
                try {
                    takePhoto();
                    runOnUiThread(() -> {
                        inference();
                    });
                    Thread.sleep(TIME_DETECTION);
                } catch (InterruptedException e) {
                    Log.e("MainActivity", "Background process error: " + e.getMessage());
                    break;
                }
            }
        });
    }


    public Interpreter loadModel(String modelPath) throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {

            // Alloca buffer per il modello
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) fileDescriptor.getLength());
            buffer.order(ByteOrder.nativeOrder());

            inputStream.getChannel().position(fileDescriptor.getStartOffset());
            inputStream.getChannel().read(buffer);
            buffer.flip();

            // Inizializza NNAPI Delegate
            nnApiDelegate = new NnApiDelegate();
            Interpreter.Options options = new Interpreter.Options();
            options.addDelegate(nnApiDelegate); // Assegna NNAPI come delegato per la GPU

            return new Interpreter(buffer, options);
        }
    }


    // Metodo per chiudere il delegate ed evitare memory leaks
    public void nnapiclose() {
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
    }


    public String processImage_8b(Bitmap bitmap) {
        // Ridimensiona e preprocessa l'immagine
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE);
        inputBuffer.order(ByteOrder.nativeOrder());

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        // Divisione del lavoro tra più thread
        for (int y = 0; y < IMAGE_SIZE; y++) {
            final int row = y;
            executor.execute(() -> {
                for (int x = 0; x < IMAGE_SIZE; x++) {
                    int pixel = resizedBitmap.getPixel(x, row);
                    int gray = (pixel >> 16) & 0xFF; // Estrai il valore in scala di grigi
                    synchronized (inputBuffer) {
                        inputBuffer.put((byte) (gray - 128)); // Centra i valori per INT8 (-128 a 127)
                    }
                }
            });
        }

        executor.shutdown();
        while (!executor.isTerminated()) { } // Attendi il completamento del processamento

        // Buffer per l'output del modello
        byte[][] output = new byte[1][EMOTIONS.length]; // Cambia in byte[][] per INT8

        // Esegui l'inferenza con TensorFlow Lite
        tflite.run(inputBuffer, output);

        // Trova l'emozione con la probabilità più alta
        int maxIndex = 0;
        byte maxProb = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        return EMOTIONS[maxIndex];
    }


    private String processImage_16f(Bitmap bitmap) {
        // Ridimensiona e preprocessa l'immagine
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 4); // float = 4 byte
        inputBuffer.order(ByteOrder.nativeOrder());

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                int gray = (pixel >> 16) & 0xFF; // Estrai il valore in scala di grigi
                inputBuffer.putFloat(gray / 255.0f); // Normalizza in [0, 1] per float16
            }
        }

        // Buffer per l'output del modello
        float[][] output = new float[1][EMOTIONS.length]; // Cambia in float[][] per float16

        // Esegui l'inferenza
        tflite.run(inputBuffer, output);

        // Trova l'emozione con la probabilità più alta
        int maxIndex = 0;
        float maxProb = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }
        return EMOTIONS[maxIndex];
    }


    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Configura il selettore per la telecamera frontale
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();

                // Configura l'anteprima
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Configura ImageCapture per scattare foto
                imageCapture = new ImageCapture.Builder().build();

                // Collega tutto al lifecycle della Activity
                Camera camera = cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageCapture
                );
            } catch (Exception e) {
                Log.e("MainActivity", "Opening camera error", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }


    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        if (rotationDegrees == 0) {
            // Flip the image horizontally without rotation
            android.graphics.Matrix matrix = new android.graphics.Matrix();
            matrix.postScale(-1, 1, bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        // Configura la matrice per la rotazione e il flip orizzontale
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(rotationDegrees);
        matrix.postScale(-1, 1, bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);

        // Crea il Bitmap ruotato e specchiato
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }


    private Bitmap imageProxyToBitmap(ImageProxy image) {
        // Ottieni il buffer dell'immagine
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        // Decodifica l'immagine in un Bitmap
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        // Applica la rotazione in base all'orientamento dell'immagine
        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        return rotateBitmap(bitmap, rotationDegrees);
    }


    private void takePhoto() {
        if (imageCapture == null) {
            Toast.makeText(this, "ImageCapture is not configured", Toast.LENGTH_SHORT).show();
            return;
        }

        imageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                Bitmap bitmap = imageProxyToBitmap(image);
                runOnUiThread(() -> {
                    if (bitmap != null) {
                        capturedBitmap = bitmap; // Assegna il Bitmap
                        imageView.setImageBitmap(capturedBitmap);
                        // Toast.makeText(MainActivity.this, "Picture displayed", Toast.LENGTH_SHORT).show();
                    }
                });
                image.close();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e("MainActivity", "Picture load error" + exception.getMessage());
            }
        });
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        nnapiclose();
    }
}
