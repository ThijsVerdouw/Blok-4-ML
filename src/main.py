import downloadData
from loguru import logger 
from torch import cuda
from torchsummary import summary
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import Trainer
import NeuralNetworks
import modelSettings

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

def createTrainer(Data: object, model:object)-> object:
    """
    The aim of this function is to combine the datastreamers and the configuration
    of the model into one pytorch 'Trainer' object. 
    Currently this does not import settings from an external class or something.
    But it needs to do that at a later stage.
    """ 
    trainLen, validLen, trainstreamer, validstreamer = createStreamers(Data)
    
    # convenient shape management:
    x, y = next(iter(trainstreamer))
    logger.info('input size = ' +str(x[1].shape))
    


    # Get cpu or gpu device for training.
    device = "cuda" if cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # Load model defined above:
    model = model.to(device)
    logger.info( summary(model, input_size= x[1].shape))

    # load the settings from the settings file:
    settings = modelSettings.trainSettings(trainLen=trainLen, validLen=validLen, model=model)

    # Put all of it in the pytorch trainer class:
    trainer = Trainer(
        model=model,
        settings= settings.trainerSettings,
        loss_fn=settings.loss_fn, #exceptionally important.
        optimizer=settings.optimizer,
        traindataloader=trainstreamer,
        validdataloader=validstreamer,
        scheduler= settings.scheduler)
    return trainer

# filters=128, units1=128, units2=64, input_size=(32, 3, 224, 224))
# x, y = next(iter(trainstreamer))
# print('input size = ' +str(x[1].shape))

ExternalModel= NeuralNetworks.CNN(filters=28*28, units1=256, units2=256, input_size=(64, 1, 28, 28))
# ExternalModel = NeuralNetworks.LLN()
Data = downloadData.downloadMnistData()
trainer = createTrainer(Data, model=ExternalModel)
trainer.loop()

# 2 linear layers:
# 2023-12-11 12:06:38.637 | INFO     | mltrainer.trainer:report:171 - Epoch 18 train 0.1603 test 0.3656 metric ['0.8885']t/s]| 18/100 [02:58<13:32,  9.91s/it] 
# 2023-12-11 12:06:38.639 | INFO     | mltrainer.trainer:__call__:214 - best loss: 0.3235, current loss 0.3656.Counter 10/10.

# 1 convo, 2 linear:
# 2023-12-11 13:24:26.544 | INFO     | mltrainer.trainer:report:171 - Epoch 64 train 0.3684 test 0.4110 metric ['0.8540']
# 100%|██████████| 1875/1875 [00:13<00:00, 134.35it/s] 81%|████████  | 81/100 [29:15<06:51, 21.67s/it]

# 3 convo, 2 linear:
# 2023-12-11 11:50:23.653 | INFO     | mltrainer.trainer:report:171 - Epoch 9 train 0.1502 test 0.2382 metric ['0.9213']
# 2023-12-11 11:50:23.653 | INFO     | mltrainer.trainer:__call__:214 - best loss: 0.2382, current loss 0.2382.Counter 3/10.
# 100%|██████████| 10/10 [24:19<00:00, 145.92s/it]

# avg pool IPV max pool kost je 10% accuracy (92% naar 82%)
# halveren units 1 en 2 (filters in lineare lagen) kost je nog eens 10% (82% naar 72%), al gaat het wel sneller.

# Dropout van 0.1 geeft 1% extra accuracy. Maakt trainen langer durend. Het zorgt in ieder geval voor overfitting en Significant snellere learning in de eerste 10 loops.
# Of geeft het meer? 

# De randomness in de eerste loop door de random weights en biasses zijn niet relevant na 10 loops.
# Dit is een big deal.

# Convolutions Groot naar klein: goed! Klein naar groot:gaat van 82% naar 57%.

# extra padding op de eerste laag geeft een accuracy verhoging van 5% (70% ->75%)



