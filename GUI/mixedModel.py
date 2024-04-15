import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import pyaudio
import wave
import librosa
from wav2vec import process_func
from wonderwords import RandomSentence


##### FUNCTIONS #####

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.tile(image, (1, 1, 1, 3))
    return image


# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_model = tf.saved_model.load('GUI/mobilenet_model')

# Initialize variables
collection_started = False
show_average_age = False
age_predictions = []

# Load the voice age prediction model
def process_audio(audio_file):
    audio_signal, sampling_rate = librosa.load(audio_file, sr=16000)
    age_prediction = process_func(audio_signal, sampling_rate)
    return age_prediction[0][0]*100 #the model returns an age from 0-100 in the format 0.XX (ie: 19 -> 0.19)

def fused_age_prediction(average_age_audio, average_age_image, audio_accuracy, image_accuracy):
    relative_image_accuracy = image_accuracy /(audio_accuracy+image_accuracy)
    relative_audio_accuracy= audio_accuracy /(audio_accuracy+image_accuracy)
    fused_age = average_age_audio*relative_audio_accuracy + average_age_image*relative_image_accuracy
    return fused_age

def age_to_ageband(age):
    rounded_age = round(age)
    age_mapping = {
        (0, 4): '0-4',
        (5, 9): '5-9',
        (10, 14): '10-14',
        (15, 19): '15-19',
        (20, 24): '20-24',
        (25, 29): '25-29',
        (30, 34): '30-34',
        (35, 39): '35-39',
        (40, 44): '40-44',
        (45, 49): '45-49',
        (50, 54): '50-54',
        (55, 59): '55-59',
        (60, 64): '60-64',
        (65, 69): '65-69',
        (70, 74): '70-74',
        (75, 79): '75-79',
        (80, 84): '80-84',
        (85, 90): '85-90'
    }
    for age_range, age_band in age_mapping.items():
        if age_range[0] <= rounded_age <= age_range[1]:
            return age_band
    return '???'

def generate_random_sentences():
    s1 = str(RandomSentence().sentence())
    s2 = str(RandomSentence().sentence())
    return s1, s2


##### MAIN #####

# Creating a VideoCapture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed!")
    exit()

# Create a window
cv2.namedWindow("Age Detection", cv2.WINDOW_GUI_NORMAL)
instruction_lines = [
    "PRESS 'S' TO START THE RECORDING",
    "PRESS 'E' TO END THE RECORDING.",
    "PRESS 'Q' TO QUIT PROGRAM."
]

while cap.isOpened():
    # Reading a frame from the webcam feed
    ret, frame = cap.read()
    if not ret:
        break
    
    #instructions on top left
    for i, line in enumerate(instruction_lines):
        cv2.putText(frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = preprocess_image(face_img)
        
        if collection_started:
            predictions = image_model(face_img)
            predicted_age = predictions[0][0]
            formatted_age = f"{predicted_age:.1f}"
            age_predictions.append(float(formatted_age))
            cv2.putText(frame, "RECORDING - ", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
        elif show_average_age:
            cv2.putText(frame, f"Predicted Age from Audio: {average_age_audio:.1f}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Predicted Age from Image: {average_age_image:.1f}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Predicted Age Band: {fused_age_band}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        else:
            cv2.putText(frame, "FACE DETECTED", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Displaying the frame
    cv2.imshow("Age Detection", frame)

    key = cv2.waitKey(1)
    if (key == ord('s') and 'stream' not in locals()):
        collection_started = True
        print("* Recording audio ...")

        # Create new window for rng sentences
        s1, s2 = generate_random_sentences()
        cv2.namedWindow("Random Sentences", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Random Sentences", 700, 500)

        white = 255 * np.ones((200, 600, 3), dtype=np.uint8)
        cv2.putText(white, "Read the following sentences during the recording:", (50, 50), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
        cv2.putText(white, s1, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(white, s2, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.imshow("Random Sentences", white)


        CHUNK = 4096  # Increased buffer size
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        audio_frames = []

        p = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            if recording:
                audio_frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=16000,  # Adjusted sampling rate for recording
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)

        recording = True

    elif (key == ord('e') and 'stream' in locals()):
        if age_predictions:
            average_age_image = sum(age_predictions) / len(age_predictions)
            filename = "output.wav"
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(44100) 
            wf.writeframes(b''.join(audio_frames))
            wf.close()
            average_age_audio = process_audio(filename)
            print(average_age_audio)
            print("Average Age from Image:", average_age_image)
            print("Age from Audio:", average_age_audio)
            fused_age = fused_age_prediction(average_age_audio, average_age_image, 55.6, 56.1)
            fused_age_band = age_to_ageband(fused_age)
            print("Fused Age Prediction:", fused_age)
            age_predictions = []
            show_average_age = True
        collection_started = False
        cv2.destroyWindow("Random Sentences")
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Close window/program if "Q" key is pressed 
    if key == ord('q') or cv2.getWindowProperty("Age Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
