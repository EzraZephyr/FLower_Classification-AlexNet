import torchvision
import torch
import pickle
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

def data_loader():

    flower_image = '../data/flower_images/'

    dataset_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                        torchvision.transforms.ToTensor(),])
    # Define a series of operations to be applied directly in image processing later
    # Resize the input images to 227x227 and convert them to tensor format

    flower_data = ImageFolder(root=flower_image, transform=dataset_transform)
    # Deploy the operations defined above to each image in the subfolders through ImageFolder
    # The folder names, which represent image categories, are converted to numerical values starting from 0
    # and stored as a dictionary

    class_to_idx = flower_data.class_to_idx
    index_to_class = {idx: cla for cla, idx in class_to_idx.items()}
    # The following is the dictionary format converted from subfolders by ImageFolder
    # {'Lilly': 0, 'Lotus': 1, 'Orchid': 2, 'Sunflower': 3, 'Tulip': 4}
    # We need to convert the values to keys so that in the prediction function later
    # we can directly output the flower names instead of numerical categories

    with open('../process_data/flower_labels.pkl', 'wb') as f:
        pickle.dump(index_to_class, f)

    X, y = [], []
    i = 0
    for image, label in flower_data:
        i += 1
        if i % 1000 == 0:
            print(i)
        # Monitor the data processing progress

        X.append(image)
        y.append(label)
        # Store the extracted image tensors and their target values

    X = torch.stack(X)
    # Stack all tensors in the list along a new dimension to form a new tensor list
    # (N, ...)

    y = torch.tensor(y)
    # Convert the target values to tensors

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the data into training and test sets

    with open('../process_data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('../process_data/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('../process_data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('../process_data/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    # Save the data