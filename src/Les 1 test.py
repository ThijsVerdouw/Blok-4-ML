from pathlib import Path
import numpy as np
from typing import Iterator, Tuple, List
import mads_datasets
mads_datasets.__version__
from mads_datasets import DatasetFactoryProvider, DatasetType
from mads_datasets.base import BaseDatastreamer
from loguru import logger

logger.info("starting")

def downloadtData ():
    """
    Download the flowers dataset and hand it to the other functions.
    """
    flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    flowersfactory.download_data()
    datasets = flowersfactory.create_dataset()
    logger.info("Train dataset contains this many records: " + str(len(datasets["train"])))
    logger.info("Test dataset contains this many records: " + str(len(datasets["valid"])))
    return datasets

def batch_processor(batch):
    """
    the batch is now a pair of (img, label) tuples. However, we want to untangle a certain amount of them into a list of images and a list of labels.
    Think of this as unzipping a zipper. Weirdly enough, in python we use the same command for this as we would use to create the pairs.
    """
    X, Y = zip(*batch)
    return np.stack(X), np.array(Y)


def collectData (train):
    """
    Dit loopt oneindig door de data heen, als je nog iets toevoegd wat de index reset als een epoch is afgerond.
    """
    datasets = collectData()
    train = datasets["train"]

    streamer = BaseDatastreamer(
        dataset=train,
        batchsize=32,
        preprocessor=batch_processor
    )

    gen = streamer.stream()
    X, y = next(gen)
    X.shape, y.shape

