import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from classes import BeeDataset_train, BeeDataset_test, NN

#======================== GLOBAL VARIABLES ============================================================
EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.001 # 10*e^-4
#====================================================================================================

# Remove all outliers that have a value of 0 and are too close to the ones that have a value of 1
def remove_outliers(data, threshold=0.5):
    data_positives = data[data['y'] == 1]
    data_negatives = data[data['y'] == 0]

    # Remove all negatives that are close to the positives by using the threshold in a 2d space of latitude and longitude
    
    # Cycle through all the positives
    for index, row in data_positives.iterrows():
        # Get the latitude and longitude of the positive
        lat = row['latitude']
        lon = row['longitude']

        # Get the euclidian distance between the positive and all the negatives
        distance = np.sqrt((data_negatives['latitude'] - lat)**2 + (data_negatives['longitude'] - lon)**2)
        
        # Remove all the negatives that are too close to the positive
        data_negatives = data_negatives[distance > threshold]
    
    print(data_negatives.shape)

    # Return the new data
    return pd.concat([data_positives, data_negatives])

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

if __name__ == "__main__":
    # Load dataset from csv file
    columnNames = ['mntcm', 'mxtwm', 'rfseas', 'tann', 'latitude', 'longitude', 'y']
    data = pd.read_csv('dataset_bees.csv', names=columnNames)

    # Remove outliers
    data = remove_outliers(data, 20.0)

    # Split dataset into features and labels
    X = data[['mntcm', 'mxtwm', 'rfseas', 'tann']]
    y = data['y']

    # Deal with missing values in dataset (replace with mean)
    X = X.fillna(X.mean())

    # Split dataset into training and testing using cross validation    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize features values to be between 0 and 1 (for better performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train.values)

    # Create datasets
    train_dataset = BeeDataset_train(X_train, y_train)
    test_dataset = BeeDataset_test(X_test)

    # Create dataloaders
    train_loader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NN().to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        
        