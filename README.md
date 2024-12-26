### Digital Systems project
# EFace

EFace is an Application for emotional face recognition developed for Android OS.

- Dataset: https://www.kaggle.com/datasets/msambare/fer2013

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.
The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
> The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).


## Project Structure
This project is divided into two main parts:
Dataset Loading and Model Training (Python)
Android Application (Kotlin)

### 1. Dataset Loading and Model Training

This part is responsible for:
   - Downloading the FER2013 dataset from Kaggle APIs.
   - Preprocessing the data (e.g., resizing, normalization).
   - Training a Convolutional Neural Network (CNN) for emotion recognition.
   - Converting the trained model to TensorFlow Lite format for Android integration.
- Key Files:
  - FEdataset.py: This script handles downloading the dataset from Kaggle using their API. It requires a Kaggle API token to be configured. It then preprocesses the data and saves it in a suitable format for training.
  - TFmodel.py: This script defines the CNN architecture, trains the model using the preprocessed data, and evaluates its performance. Finally, it converts the trained model to TensorFlow Lite format and saves it as model.tflite. This file will be integrated into the Android application.

### 2. Android Application

This part is responsible for:
- Integrating the TensorFlow Lite model (model.tflite).
- Capturing camera frames or loading images.
- Preprocessing input images for the model.
- Running inference using the model.
- Displaying the predicted emotion.
- Development:
  - The Android application is developed using Android Studio and Kotlin. It will utilize the CameraX API for camera access and TensorFlow Lite for inference.
  
- Future Work:
  - Implement real-time emotion recognition using the device's camera.
  Improve the user interface and add features like saving recognized emotions.
  Explore different CNN architectures and optimization techniques to enhance accuracy.

## Getting Started
1. Dataset and Model Training:
Install the required Python libraries: pip install tensorflow keras pandas numpy kaggle
Configure your Kaggle API token.
Run FEdataset.py to download and preprocess the dataset.
Run TFmodel.py to train the model and generate model.tflite.
2. Android Application:
Open the Android project in Android Studio.
Place the model.tflite file in the assets folder of the Android project.
Implement the remaining Android application logic.