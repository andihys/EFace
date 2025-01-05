package com.example.emotionalfaces;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;
    private ImageView imageView;
    private TextView textView;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 10;

    private Bitmap capturedBitmap;
    private Interpreter tflite;

    private static final String MODEL_PATH = "emotion_model.tflite";
    private static final int IMAGE_SIZE = 48; // Dimensione richiesta dal modello
    private static final String[] EMOTIONS = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"};


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

        // Aggiungi azione per il pulsante "Take Picture"
        buttonTakePicture.setOnClickListener(v -> takePhoto());

        // Azione per processare l'immagine
        buttonProcess.setOnClickListener(v -> {
            if (capturedBitmap != null) {
                String emotion = processImage(capturedBitmap);
                textView.setText(getString(R.string.emotion_detected) + emotion);
            } else {
                Toast.makeText(this, "Take a picture!", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private Interpreter loadModel(String modelPath) throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {

            ByteBuffer buffer = ByteBuffer.allocateDirect((int) fileDescriptor.getLength());
            buffer.order(ByteOrder.nativeOrder());

            inputStream.getChannel().position(fileDescriptor.getStartOffset());
            inputStream.getChannel().read(buffer);
            buffer.flip();

            return new Interpreter(buffer);
        }
    }

    private String processImage(Bitmap bitmap) {
        // Ridimensiona e preprocessa l'immagine
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE);
        inputBuffer.order(ByteOrder.nativeOrder());

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                int gray = (pixel >> 16) & 0xFF; // Estrai il valore in scala di grigi
                inputBuffer.put((byte) (gray - 128)); // Centra i valori per INT8 (-128 a 127)
            }
        }

        // Buffer per l'output del modello
        byte[][] output = new byte[1][EMOTIONS.length]; // Cambia in byte[][] per INT8

        // Esegui l'inferenza
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
            return bitmap; // Nessuna rotazione necessaria
        }

        // Configura la matrice per la rotazione
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(rotationDegrees);

        // Crea il Bitmap ruotato
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
                        Toast.makeText(MainActivity.this, "Picture displayed", Toast.LENGTH_SHORT).show();
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
                Toast.makeText(this, "Permesso per la fotocamera negato", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}
