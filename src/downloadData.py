import torch 
from mads_datasets import DatasetFactoryProvider, DatasetType
from loguru import logger 


def downloadMnistData():
    """
    Input: none
    returns: MADS datafactory.
    Downloads a dataset using an existing library
    """
    fashion_factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    fashion_factory.download_data()
    logger.info('Data downloaded and stored in: ' + str(fashion_factory.subfolder))
    
    return fashion_factory


def createDataset(Data: object):
    """
    test -> dataset
    train -> dataset
    The mads dataset have pre-existing labels. They have already split the datset in a train and test. 
    All I have to do is copy it over and select the correct labels
    """
    Data = Data.create_dataset()
    train = Data['train']
    # print(Data)
    test = Data['valid']
    logger.info('train test split: ' + str(len(train)) + ' ' + str(len(test)))
    return train, test 

