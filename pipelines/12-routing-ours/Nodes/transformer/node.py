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

    def transform_input(self, features, names=[], meta=[]):
        return self.predict(features, names, meta)

    def transform_output(self, features, names=[], meta=[]):
        return self.predict(features, names, meta)
