import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AgePredictionModel(nn.Module):
    def __init__(self, l1_lambda=0.01, l2_lambda=0.01):
        super(AgePredictionModel, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 512)
        # self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        # print("Weight matrix data type:", self.fc1.weight.dtype)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # x = self.dropout(x)
        x = self.fc7(x) # No activation, direct regression
        
        # # L1 regularization
        # l1_reg = torch.tensor(0., device=x.device)
        # for param in self.parameters():
        #     l1_reg += torch.norm(param, 1)

        # # L2 regularization
        # l2_reg = torch.tensor(0., device=x.device)
        # for param in self.parameters():
        #     l2_reg += torch.norm(param, 2)

        # return x, self.l1_lambda * l1_reg + self.l2_lambda * l2_reg 
        return x