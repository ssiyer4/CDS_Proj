import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import pyaudio
import wave
import librosa
from wav2vec import process_func

##### FUNCTIONS #####

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.tile(image, (1, 1, 1, 3))
    return image



##### MAIN #####

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.saved_model.load('GUI/mobilenet_model')

# Creating a VideoCapture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed!")
    exit()

# Create a window
cv2.namedWindow("Age Detection", cv2.WINDOW_GUI_NORMAL)

# Initialize variables
collection_started = False
show_average_age = False  # Flag to indicate whether to display the average age or not
age_predictions = []

while cap.isOpened():
    # Reading a frame from the webcam feed
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = preprocess_image(face_img)
        
        # Predict age
        if collection_started:
            predictions = model(face_img)
            predicted_age = predictions[0][0]
            formatted_age = f"{predicted_age:.1f}"
            age_predictions.append(float(formatted_age))
            cv2.putText(frame, "RECORDING", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif show_average_age:
            cv2.putText(frame, f"Age: {average_age:.1f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "PRESS 'S' TO START", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Displaying the frame
    cv2.imshow("Age Detection", frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('s'):
        collection_started = True
        show_average_age = False  # Reset the flag when starting a new collection
    elif key == ord('e'):
        if age_predictions:
            average_age = sum(age_predictions) / len(age_predictions)
            print("Average Predicted Age:", average_age)
            age_predictions = []
            show_average_age = True  # Set the flag to display the average age
        collection_started = False

    # Close window/program if "Q" key is pressed 
    if key == ord('q') or cv2.getWindowProperty("Age Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
