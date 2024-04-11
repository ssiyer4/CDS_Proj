from torch.utils.data import DataLoader
from data.voice_dataset import VoiceDataset
from utils.accuracy import calculate_accuracy, map_to_age_band
from models.cnn_model import AgePredictionModel
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import argparse
import mlflow
import torch

def train_one_epoch(voice_model, train_data, optimizer, criterion) -> None:
    """
    Trains CNN for one epoch, for age prediction
    :param voice_model: CNN model
    :param train_data: training data
    :param optimizer: optimizer
    :param criterion: loss criterion
    """
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
 
    voice_model.train()
 
    for voice_fts, targets in tqdm(train_data):
        
        targets = targets.to(torch.float32)

        optimizer.zero_grad()
        outputs = voice_model(voice_fts)
        outputs = outputs.to(torch.float32)
        mse_loss = criterion(outputs.squeeze(), targets.float())
        rmse_loss = torch.sqrt(mse_loss) # Compute the RMSE from MSE
        # total_loss = rmse_loss + reg_term
        rmse_loss.backward()
        optimizer.step()
        print(rmse_loss.item())
 
        mlflow.log_metric("train_loss", rmse_loss.item())
 
def evaluate_one_epoch(voice_model, val_data, epoch, criterion):
    """
    Evaluates CNN for one epoch, for age prediction
    :param voice_model: CNN model
    :param train_data: validation data
    :param optimizer: optimizer
    :param criterion: loss criterion
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    voice_model.eval()

    with torch.no_grad():
        for voice_fts, targets in tqdm(val_data):

            targets = targets.to(torch.float32)

            preds = voice_model(voice_fts)
            preds = preds.to(torch.float32)
            mse_loss = criterion(preds.squeeze(), targets.float())
            rmse_loss = torch.sqrt(mse_loss)
            print(rmse_loss.item())

            mlflow.log_metric("val_loss", rmse_loss.item())
 
 
def test_voice_model(voice_model, test_data, criterion):
    """
    Tests CNN for one epoch, for age prediction
    :param voice_model: CNN model
    :param train_data: test data
    :param optimizer: optimizer
    :param criterion: loss criterion
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voice_model.eval()

    acc = []

    with torch.no_grad():
        for voice_fts, targets in tqdm(test_data):

            targets = targets.to(torch.float32)

            preds = voice_model(voice_fts)
            preds = preds.to(torch.float32)
            mse_loss = criterion(preds.squeeze(), targets.float())
            rmse_loss = torch.sqrt(mse_loss)
            print(rmse_loss.item())
            
            batch_acc = calculate_accuracy(preds.squeeze().detach().numpy().tolist(), targets.detach().numpy().tolist(), 6)

            acc.append(batch_acc)

            mlflow.log_metric("test_loss", rmse_loss.item())
    
    final_acc = np.average(acc)
    mlflow.log_metric("final_accuracy", final_acc)
 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # experiment_id = mlflow.create_experiment("Training with validation set")
    with mlflow.start_run(experiment_id="456957762126250708"):
        # Data
        voice_dataset = VoiceDataset(args.meta)
        train_data_tmp, val_data_tmp, test_data_tmp = torch.utils.data.random_split(voice_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
        # train_data_tmp, test_data_tmp = torch.utils.data.random_split(voice_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

        criterion = nn.MSELoss()

        train_data = DataLoader(
            train_data_tmp,
            batch_size=args.batch_size,
            shuffle=False,
            # collate_fn=lambda x: tuple(zip(*x)),
        )
        val_data = DataLoader(
            val_data_tmp,
            batch_size=args.batch_size,
            shuffle=False,
            # collate_fn=lambda x: tuple(zip(*x)),
        )
        test_data = DataLoader(
            test_data_tmp,
            batch_size=args.batch_size,
            shuffle=False,
            # collate_fn=lambda x: tuple(zip(*x)),
        )
        print("train_data:", train_data)
        print("val_data:", val_data)
        print("test_data:", test_data)
 
        # Pre-trained Faster R-CNN model
        voice_model = AgePredictionModel()
 
        params = voice_model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=0.0005
        )
 
        # Training Loop
        for epoch in range(args.epochs):
            print("training epoch:", epoch)
            train_one_epoch(voice_model, train_data, optimizer, criterion)
            print("evaluating epoch:", epoch)
            evaluate_one_epoch(voice_model, val_data, epoch, criterion)
            pass
        
        mlflow.log_metric("lr", args.lr)
        mlflow.log_metric("epochs", args.epochs)

        torch.save(voice_model, args.save_as)
 
    # Testing Loop
    with mlflow.start_run(experiment_id="456957762126250708"):
        pass
        voice_model_trained = torch.load(args.save_as, map_location=torch.device('cpu'))
        voice_model_trained.to(device)
        print("testing model: ", args.save_as)
        test_voice_model(voice_model_trained, test_data, criterion)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--meta", type=str, default="data/FINAL_AUDIO_FEATURES_19916.csv", required=False)
    parser.add_argument("--lr", type=float, default=0.0065, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--save_as", type=str, required=True)
    # parser.add_argument("--", type=, default=)
    args = parser.parse_args()
    main(args)