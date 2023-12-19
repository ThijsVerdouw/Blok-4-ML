from mltrainer import TrainerSettings, ReportTypes, metrics
from pathlib import Path
import torch.optim as optim
from torch import nn

class trainSettings ():
    def __init__(self, trainLen:int, validLen:int, model:object):
        """
        Needed to use the init because I actually need the train and valid len.    
        """
        self.trainerSettings = TrainerSettings(
            epochs=100,
            metrics=[metrics.Accuracy()],
            logdir=Path("logs").absolute(),
            train_steps=trainLen,
            valid_steps=validLen,
            reporttypes=[ReportTypes.TENSORBOARD],
        ) 
        self.loss_fn=nn.CrossEntropyLoss() #exceptionally important.
        self.optimizer= optim.Adam
        # self.scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9), patience=5)
        self.scheduler= optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler.patience = 5 #werk niet.
        # self.batchsize = 32


class settings ():
    dropout = nn.Dropout(0.1)
# setting = trainSettings(1,2)
# print(trainSettings.settings)