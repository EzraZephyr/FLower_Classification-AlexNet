import pickle

import torchvision
import torch
from PIL import Image
from utils_en.model import Flower

def predict_upload(image_path):

    with open('../process_data/flower_labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    file = image_path
    # If you don't want to use GUI, you can change image_path to the address of the image

    image = Image.open(file).convert('RGB')
    # Load the image from the path and convert it to RGB format

    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Resize((227, 227))])
    # Define a series of operations to be applied in image processing later
    # Resize the input image to 227x227 and convert it to tensor format

    image = image_transform(image)
    # Since there is only one image and no labels, simply convert the format without using ImageFolder

    image = image.unsqueeze(0)
    # Add a dimension at the front because the input format requires a batch size
    # [1, 3, 227, 227]

    model = Flower(3, 5)
    model.load_state_dict(torch.load('../model/flower_model_30.pt', map_location=torch.device('cpu')))
    model.eval()
    # Build the model, load the model, and set to evaluation mode

    with torch.no_grad():
        # Disable gradient computation

        output = model(image)
        # Forward pass

        _, predicted = torch.max(output, 1)
        # Return the index of the maximum probability among the five categories

        print(f'predicted: {labels[predicted.item()]}')

        return labels[predicted.item()]
        # Retrieve the final answer from the dictionary saved in the data_loader file and return it
