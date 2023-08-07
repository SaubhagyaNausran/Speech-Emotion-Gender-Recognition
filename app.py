import os
import sys
import random
import librosa
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json

# Create Flask app
app = Flask(__name__)

Features = pd.read_csv('New_Features.csv')
X = Features.iloc[:, :-1].values
Y = Features['labels'].values
encoder = OneHotEncoder()
Y_res = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, Y_res, test_size=0.2, random_state=42, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# making our data compatible with the model
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# loading json and model architecture
json_file = open('Emotion_Model_conv1d_gender_93.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into the new model
loaded_model.load_weights('Emotion_Model_conv1d_gender_93.h5')
print("Loaded model from disk")

# Keras optimizer
loaded_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess audio and perform prediction
def predict_emotion_and_gender(audio_file_path):
    # Load the audio file
    data, sample_rate = librosa.load(audio_file_path)

    # Extract features from the audio data
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    features = np.array(mfccs_processed)

    # Reshape the features for model input
    features = np.reshape(features, (1, features.shape[0], 1))

    # Perform the predictions
    predictions = loaded_model.predict(features)

    # Decode the predicted labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    gender_labels = ['Male', 'Female']
    emotion_gender=np.argmax(predictions[0, :])
    if emotion_gender<=6:
        predicted_gender=gender_labels[1]
    else:
        predicted_gender = gender_labels[0]
    predicted_emotion = emotion_labels[np.argmax(predictions[0, :])%7]
    print(predictions)
    return predicted_emotion, predicted_gender

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file found'})

    audio = request.files['audio_file']

    # Save the audio file
    audio_path = os.path.join(os.getcwd(), 'audio.wav')
    audio.save(audio_path)

    # Perform the emotion and gender prediction
    predicted_emotion, predicted_gender = predict_emotion_and_gender(audio_path)

    return jsonify({'emotion': predicted_emotion, 'gender': predicted_gender})

if __name__ == '__main__':
    app.run(debug=True)
