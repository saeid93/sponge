import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class NodeOne(object):
    def predict(self, X, features_names=None):
        logger.info(f"incoming input:\n{X}\nserved!")
        return X*2
