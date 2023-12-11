import downloadData
from loguru import logger 
import torch
from torch import nn
  import torch.optim as optim
from torchsummary import summary
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import metrics
from mltrainer import TrainerSettings, ReportTypes, Trainer
from pathlib import Path

log_dir=Path("logs").absolute()
Data = downloadData.downloadMnistData()

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Basic neural network with flatten -> 3 hidden layers
        """
        super(NeuralNetwork, self).__init__()
        # images are not 2d data, and linear regression requires 2d data.
        # That's why flatten is here.
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(786, 512),
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


def createStreamers (Data: object) -> tuple[int, int, object, object]:
    """
    This collects the downloaded mads dataset and turns it into streamers
    using the inbuilt preprocesser in pytorch. Which is very fast.
    it also gets the number of records in each category for the Trainer.
    """
    # batchsize for streamer is 64, using the pytorch preprocesser.
    streamers = Data.create_datastreamer(batchsize=64, preprocessor=BasePreprocessor() )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return len(train), len(valid), trainstreamer, validstreamer

def createTrainer(Data: object)-> object:
    """
    The aim of this function is to combine the datastreamers and the configuration
    of the model into one pytorch 'Trainer' object. 
    Currently this does not import settings from an external class or something.
    But it needs to do that at a later stage.
    """ 
    trainLen, validLen, trainstreamer, validstreamer = createStreamers(Data)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # Load model defined above:
    model = NeuralNetwork().to(device)
    logger.info( summary(model, input_size=(1, 28, 28)))

    # Set loss function (very important)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set settings, normally done somewhere else I think?
    settings = TrainerSettings(
        epochs=10,
        metrics=[metrics.Accuracy()],
        logdir=log_dir,
        train_steps=trainLen,
        valid_steps=validLen,
        reporttypes=[ReportTypes.TENSORBOARD],
    )

    # Put all of it in the pytorch trainer class:
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=trainstreamer,
        validdataloader=validstreamer,
        scheduler= optim.lr_scheduler.ReduceLROnPlateau)
    return trainer
trainer = createTrainer(Data)
trainer.loop()