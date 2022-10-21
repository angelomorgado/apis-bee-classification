import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#======================== Class that represents the dataset
# This class is used to create a training dataset that can be used by the DataLoader class
class BeeDataset_train(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# This class is used to create a testing dataset that can be used by the DataLoader class
class BeeDataset_test(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

#====================== Class that represents the neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # Layers
        self.inputLayer = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 16)
        self.layer3 = nn.Linear(16, 8)
        self.outputLayer = nn.Linear(8, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.outputLayer(x)
        x = self.sigmoid(x)
        return x

# Explanation of the code:

# This neural network has 4 input neurons, 
# 8 neurons in the first hidden layer, 
# 16 neurons in the second hidden layer, 
# 8 neurons in the third hidden layer 
# and 1 output neuron.
# In total there are 1 input layer, 3 hidden layers and 1 output layer.