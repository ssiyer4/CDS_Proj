import numpy as np
import librosa
import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class ModelHead(nn.Module):
    """Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeModel(Wav2Vec2PreTrainedModel):
    """Speech age classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)  # Predicting a single value for age
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        return logits_age

# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeModel.from_pretrained(model_name).to(device)

# dummy signal
sampling_rate = 16000
# signal, sr = librosa.load("/Users/shwethaiyer/Downloads/CDS_Project/data/audio/accent_kaggle_twi5.flac", sr=sampling_rate)

def process_func(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    # Normalize and prepare signal
    inputs = processor(signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Move input to the correct device
    input_values = inputs.input_values.to(device)

    # Ensure the input is 2D: [batch_size, sequence_length]
    if input_values.ndim == 3:
        input_values = input_values.squeeze(0)

    # Forward pass through the model
    with torch.no_grad():
        logits_age = model(input_values)

    # Convert logits to numpy for further processing or analysis
    age_prediction = logits_age.detach().cpu().numpy()
    return age_prediction


# Example usage
# age_prediction = process_func(signal, sampling_rate)
# print(age_prediction)

data = pd.read_csv("data/FINAL_AUDIO_FEATURES_19916.csv")

list_audio = os.listdir('/Users/shwethaiyer/Downloads/CDS_Project/data/audio')

for i, audio in enumerate(list_audio):
    print(i)
    signal, sr = librosa.load(f"/Users/shwethaiyer/Downloads/CDS_Project/data/audio/{audio}", sr=sampling_rate)

    duration = librosa.get_duration(y=signal, sr=sampling_rate)

    age_prediction = process_func(signal, sampling_rate)

    real_age = int(data.loc[data['filename'] == audio]['age'])

    with open("/Users/shwethaiyer/Downloads/CDS_Project/data/wav2vec2result3.txt", "a") as file:
        file.write(f"{audio}\t{duration}\t{real_age}\t{age_prediction[0][0]}\t\n")
