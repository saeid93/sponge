import logging
from textwrap import indent
import torch
from copy import deepcopy
import os
import numpy as np
from transformers import pipeline
from seldon_core.user_model import SeldonResponse

logger = logging.getLogger(__name__)


class GeneralNLP(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logging.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'distilbert-base-uncased-finetuned-sst-2-english'
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logging.info(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.MODEL_VARIANT = 'sentiment-analysis'
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")

    def load(self):
        logger.info('Loading the ML models')
        self.model  = pipeline(task=self.TASK, model=self.MODEL_VARIANT)
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f'Incoming input type: {type(X)}')
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        output = self.model(X['text'])
        output = output[0]
        logger.info(f"Output:\n{output}\nwas sent!")
        return X
