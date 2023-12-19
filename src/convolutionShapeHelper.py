import torch
import torch.nn as nn
from pathlib import Path
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor

preprocessor = BasePreprocessor()

def getData():
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
    # flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    # streamers = flowersfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]

    trainstreamer = train.stream()
    validstreamer = valid.stream()
    x, y = next(iter(trainstreamer))
    print('input size = ' +str(x[1].shape))
    return x,y

x,y = getData()

# de in channels voor iedere foto in de batch is:
in_channels = x.shape[1]

# this is to play around with one convolution:
# conv = nn.Conv2d(
#     in_channels=in_channels,
#     out_channels=64,
#     kernel_size=10,
#     padding=(1,1))
# out = conv(x)
# out.shape

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 128, kernel_size=7, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    # nn.MaxPool2d(kernel_size=1),
    # nn.Flatten(),
)
out = convolutions(x)
print(out.shape)