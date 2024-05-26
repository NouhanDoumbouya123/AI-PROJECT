from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import logging
import os

app = Flask(__name__)

# Constants
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST = ["NonViolence", "Violence"]

# Load your existing model
model = load_model('ViolenceDetectionModel.h5', compile=False)

logging.basicConfig(level=logging.DEBUG)

def preprocess_video(video_file_path, sequence_length=SEQUENCE_LENGTH, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    frames_list = []

    cap = cv2.VideoCapture(video_file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0

        frames_list.append(frame)

    cap.release()

    num_frames = len(frames_list)
    if num_frames < sequence_length:
        frames_list.extend([frames_list[-1]] * (sequence_length - num_frames))
    elif num_frames > sequence_length:
        frames_list = frames_list[:sequence_length]

    frames_array = np.array(frames_list)

    return frames_array

def predict_violence(video_file_path):
    frames_array = preprocess_video(video_file_path)
    if len(frames_array) == 0:
        logging.error("Failed to preprocess video. frames_list is empty.")
        return "Unknown"

    predicted_labels_probabilities = model.predict(np.expand_dims(frames_array, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]

    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            os.makedirs('uploads', exist_ok=True)
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    if filename.endswith(('.mp4', '.avi', '.mov')):  # Check if filename ends with video file extensions
        video_file_path = f"uploads/{filename}"
        predicted_class_name = predict_violence(video_file_path)
        return render_template('predict.html', filename=filename, prediction_result=predicted_class_name)
    else:
        return "Invalid file format. Please upload a supported video file."

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
