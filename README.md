# Multimodal Age Prediction

## Overview
This repository contains code for the 50.038 Computational Data Science course project, focused on Multimodal Age Prediction from Face and Voice input.

## Models & Data
All model explorations can be found in the respective folders `Image Models` and `Audio Models`. Within each of these folders, there is also a `Data` subfolder in which the CSVs for the sampled data can be found. Original datasets can be found referenced and credited in our final report.


## GUI

Our GUI was implemented through a simple python script that takes in video feed and seperates frames and audio, as per our model design. Audio was processed with pyaudio, and face detection was handled by Haar Cascade model. 

Through simple key presses, the user can start and end the recording. The user is prompted to repeat sentences during the recording, these sentences are generated with wonderwords library.


To run the GUI with the final model, follow these steps:

1. Navigate to the `GUI` directory.
2. Run `mixedModels.py`.
