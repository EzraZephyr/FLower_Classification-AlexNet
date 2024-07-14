from torch import nn

class Flower(nn.Module):

    def __init__(self, in_dim, n_class):
        super(Flower, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 96, 11, 4, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            # Define the convolutional and pooling layers in the AlexNet method, including ReLU functions
            # You can search online for "AlexNet architecture" to understand in detail

        )

        self.fc = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class),
            # Define two fully connected layers and an output layer, including ReLU functions
            # You can search online for "AlexNet architecture" to understand in detail

        )

    def forward(self, input):
        X = self.conv(input)
        # Pass the input data through the defined convolutional and pooling layers

        X = X.view(X.size(0), -1)
        # Flatten the resulting data by multiplying the second, third, and fourth dimensions, preparing it for the fully connected layers

        X = self.fc(X)
        # Pass the data through the fully connected layers

        return X
