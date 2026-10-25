import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A small CNN for 28x28 grayscale MNIST digits.

    Shape of the data as it flows through the network:
        input            -> 1 x 28 x 28   (1 channel = grayscale)
        conv1 + ReLU     -> 32 x 28 x 28
        maxpool          -> 32 x 14 x 14
        conv2 + ReLU     -> 64 x 14 x 14
        maxpool          -> 64 x 7 x 7
        flatten          -> 3136
        fc1 + ReLU       -> 128
        dropout
        fc2              -> 10            (one score per digit 0-9)
    """

    def __init__(self):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
