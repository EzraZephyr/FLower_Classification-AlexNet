import pickle
import torch
from torch.utils.data import DataLoader

from utils_en.model import Flower
from utils_en.data_process import data_process

def predict():
    with open('../process_data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('../process_data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    test_process = data_process(X_test, y_test)
    # Create a data processing object

    data_loader = DataLoader(test_process, batch_size=1, shuffle=False)
    # Create a data loader

    model = Flower(3, 5)
    # Build the model

    model.load_state_dict(torch.load('../model/flower_model_30.pt', map_location=torch.device('cpu')))
    # Load the model
    # Since the model was trained using CUDA, if testing on a non-CUDA device, map_location needs to be used to move it to CPU

    model.eval()
    # Set to evaluation mode

    total = 0
    correct = 0

    with torch.no_grad():
        # Disable gradient computation

        for X, y in data_loader:
            output = model(X)
            # Forward pass

            _, pred = torch.max(output, 1)
            # Return the maximum value and its index among the five categories

            total += y.size(0)
            # Count the total number of samples

            correct += (pred == y).sum().item()
            # Count the number of correctly predicted samples

    print(f'Accuracy: {100 * correct / total}%')

