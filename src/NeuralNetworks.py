import torch
from torch import nn
from loguru import logger 
import modelSettings

settings = modelSettings.settings()

# Define model
class CNN(nn.Module):
    def __init__(self, filters: int, units1: int, units2: int, input_size: tuple):
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.expansion = 2

        if True == False:
            self.convolutions = nn.Sequential(
                settings.dropout,
                nn.Conv2d(self.in_channels, self.in_channels * self.expansion, kernel_size=7, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                # nn.AvgPool2d(kernel_size=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=1), #this layer is a pseudo duplicate I think, as the main model uses an avgpool to slap it all down.
            )
        else: 
            self.convolutions = nn.Sequential(
                settings.dropout,
                nn.Conv2d(self.in_channels, self.in_channels * self.expansion, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                # nn.AvgPool2d(kernel_size=2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=1), #this layer is a pseudo duplicate I think, as the main model uses an avgpool to slap it all down.
            )

        activation_map_size = self._conv_test(self.input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.expansion, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, 10)
        )

    def _conv_test(self, input_size:int):
        x = torch.ones(input_size, dtype=torch.float32)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits



class LLN(nn.Module):
    def __init__(self):
        """
        Basic neural network with flatten -> 3 hidden layers.
        Do not understand why this does not have the same input values
        """
        super(LLN, self).__init__()
        # images are not 2d data, and linear regression requires 2d data.
        # That's why flatten is here.
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            settings.dropout,
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        """
        This runs the neural network described above.
        This one does the following:
        1. flatten the pictures
        2. run the linear layers
        3. Return logits (recommendations, sort of)
        """
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

a = LLN()