import pyaudio
import numpy as np
import wave
import librosa
from wav2vec import process_func

def record_audio(filename, samplerate):
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
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    print("* Press 's' to start recording, 'e' to end recording")

    recording = False

    while True:
        key = input()
        if key == 's' and not recording:
            print("* Recording audio...")
            recording = True
            stream.start_stream()
        elif key == 'e' and recording:
            print("* Finished recording")
            recording = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            break

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

    print("* Audio saved as '{}'".format(filename))

    return filename

def process_audio(audio_file):
    audio_signal, sampling_rate = librosa.load(audio_file, sr=16000)  # adjust sampling rate here
    age_prediction = process_func(audio_signal, sampling_rate)
    print("Predicted age:", age_prediction)


if __name__ == "__main__":
    OUTPUT_FILENAME = "output.wav"
    SAMPLERATE = 44100  # Adjusted sample rate for MacBook internal microphone or AirPods

    recorded_audio_file = record_audio(OUTPUT_FILENAME, SAMPLERATE)
    process_audio(recorded_audio_file)
