import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Constants
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64     
CLASSES_LIST = ["NonViolence", "Violence"]
DATASET_DIR = "dataset"

# Function to extract frames from videos
def extract_frames(video_file_path):
    frames = []
    cap = cv2.VideoCapture(video_file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frames.append(resized_frame)
    cap.release()
    return frames

# Load dataset
X = []
y = []
for class_name in CLASSES_LIST:
    class_dir = os.path.join(DATASET_DIR, class_name)
    for video_file in os.listdir(class_dir):
        video_file_path = os.path.join(class_dir, video_file)
        frames = extract_frames(video_file_path)
        if len(frames) >= SEQUENCE_LENGTH:
            frames = frames[:SEQUENCE_LENGTH]  # Truncate to ensure fixed sequence length
        else:
            # Pad with black frames if shorter
            padding = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)] * (SEQUENCE_LENGTH - len(frames))
            frames.extend(padding)
        X.append(frames)
        y.append(CLASSES_LIST.index(class_name))

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess frames
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Reshape the input data to match the LSTM input shape
X_train = X_train.reshape(X_train.shape[0], SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
X_test = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(CLASSES_LIST))
y_test = to_categorical(y_test, num_classes=len(CLASSES_LIST))

# Define the model
model = Sequential([
    TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'), input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES_LIST), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('ViolenceDetectionModel.h5')
