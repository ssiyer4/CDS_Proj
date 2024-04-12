import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.tile(image, (1, 1, 1, 3))
    return image

# Load the pre-trained models - Haar Cascade classifier for face detection + mobilenet (w transfer learning) for age detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.saved_model.load('mobilenet_model')

# Creating a VideoCapture object
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Unable to read camera feed!")
    exit()

# create a window
cv2.namedWindow("Age Detection", cv2.WINDOW_GUI_NORMAL)

while cap.isOpened():
    # Reading a frame from the webcam feed
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # process each detected face
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img=preprocess_image(face_img)
        
        # predict age
        predictions = model(face_img)
        predicted_age = predictions[0][0]
        
        # display predicted age above the bounding box
        cv2.putText(frame, f"Age: {predicted_age:.1f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Displaying the frame
    cv2.imshow("Age Detection", frame)

    # close window/program if "Q" key is pressed 
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty("Age Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
