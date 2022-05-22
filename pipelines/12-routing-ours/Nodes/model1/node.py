import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class Node:
    def predict(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        return features.tolist()

    def init_metadata(self):
        meta = {
            "name": "model-one"
        }
        return meta
