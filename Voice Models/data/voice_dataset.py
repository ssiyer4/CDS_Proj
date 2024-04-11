import pandas as pd
import torch

class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 30]
        audio_fts = torch.tensor(self.data.iloc[idx, [x for x in range(0,30)]].tolist(), dtype=torch.float32)       

        target = self.data.iloc[idx, 31]

        return audio_fts, target