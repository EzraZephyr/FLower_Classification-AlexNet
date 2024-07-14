import pickle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils_en.data_process import data_process
from utils_en.model import Flower

def train():

    with open("../process_data/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open("../process_data/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    train_process = data_process(X_train, y_train)
    # Create a data processing object

    data = DataLoader(train_process, batch_size=32, shuffle=True)
    # Create a data loader with batch size of 32 and shuffle the data
    # It will automatically calculate how many batches are needed based on batch size

    model = Flower(3, 5)
    # Initialize the model with input dimension as 3 because an image has three channels
    # and output classes as 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # If a CUDA device is available, run on CUDA, otherwise use CPU

    criterion = nn.CrossEntropyLoss()
    # Define the cross-entropy loss function

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use the Adam optimizer

    epochs = 30
    # Set the number of training epochs

    train_log = '../model/training.log'
    file = open(train_log, 'w')
    # Open a file to print training logs (good practice)

    for epoch in range(epochs):

        epoch_idx = 0
        total_loss = 0
        start_time = time.time()
        # Record the training epoch index, total loss, and start time

        for X, y in data:
            X, y = X.to(device), y.to(device)
            # Transfer the input data to the specified device

            output = model(X)
            # Forward pass

            optimizer.zero_grad()
            # Zero the gradients

            loss = criterion(output, y)
            # Compute the loss

            loss.backward()
            # Backward pass

            optimizer.step()
            # Update the model parameters

            total_loss += loss.item()
            # Accumulate the loss for each batch

            epoch_idx += 1

        message = 'Epoch:{}, Loss:{:.4f}, Time:{:.2f}'.format(epoch, total_loss / epoch_idx, time.time() - start_time)
        file.write(message + '\n')
        # Store each epoch's training information in the log

        print(message)

    file.close()
    # Close the log file

    torch.save(model.state_dict(), '../model/flower_model_30.pt')
    # Save the model
