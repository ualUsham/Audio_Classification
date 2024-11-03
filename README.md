# Audio Classification using MFCC Features

## Overview

This repository contains a machine learning project aimed at classifying urban sound audio files into 10 different classes using Mel-frequency cepstral coefficients (MFCC) as features. The model is built using TensorFlow and Keras, and it is trained on the UrbanSound8K dataset, which consists of various urban sounds collected from diverse environments.

## Dataset

The project utilizes the **UrbanSound8K** dataset, which includes 8,732 audio clips categorized into 10 classes, such as:
- Air Conditioners
- Car Horns
- Children Playing
- Dog Barks
- Gunshots
- Sirens
- Street Music
- Engine Idling
- Jackhammer
- Water Running

## Features Extraction

Audio features are extracted using the **Librosa** library, specifically employing MFCCs, which capture the essential characteristics of audio signals while minimizing the dimensionality of the data.

### Feature Extraction Function

The `extract_features` function loads audio files and computes the MFCC features, returning an averaged array of these features for each audio clip.

## Model Architecture

The model architecture is designed using Keras and consists of:

- An input layer accepting MFCC features (shape: 50)
- Three hidden layers to address its complexity of the problem
- First hidden layer with 128 neurons and ReLU activation
- Second hidden layer with 64 neurons and ReLU activation
- Third hidden layer with 32 neurons and ReLU activation
- An output layer with Softmax activation to predict one of the 10 classes.
- A total of 15,434 parameters were trained based on the complexity of the problem.

## Training

The model is compiled using categorical crossentropy loss and Adam optimizer, and it is trained for 100 epochs with a batch size of 32. Validation is performed using a portion of the training data.

## Results

The model achieved an impressive **test accuracy of approximately 88.67%**, indicating its effectiveness in classifying urban sounds accurately.

## Usage

To use this model for audio classification:

1. **Install Dependencies**: Ensure you have the necessary libraries:

   ```bash
   pip install numpy pandas librosa tensorflow keras tqdm
   ```

2. **Load the Model**:

   ```python
   from keras.models import load_model
   loaded_model = load_model('model.keras')
   ```

3. **Predict Class for an Audio File**:

   Use the following code to classify a new audio file:

   ```python
   import IPython.display as ipd
   file_path = 'path_to_your_audio_file.wav'  # Update this path
   file_data = extract_features(file_path)
   x_predict = file_data.reshape(1, -1)
   y_predict = loaded_model.predict(x_predict)
   predicted_class_label = np.argmax(y_predict)
   predicted_class = labelencoder.inverse_transform([predicted_class_label])
   print(predicted_class)
   ipd.Audio(file_path)
   ```

## Requirements
To run this project, you will need the following Python libraries:

- NumPy
- Pandas
- Librosa
- TensorFlow/Keras
- IPython
- OS
## Conclusion

This project demonstrates the application of machine learning techniques in audio classification. The achieved accuracy showcases the model's potential for practical implementations in sound recognition tasks.
