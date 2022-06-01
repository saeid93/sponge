import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class VideoCombiner:
    def aggregate(self, features, names=[], meta=[]):
        logger.info(f"input type: {type(features)}")
        logger.info(f"input element type: {type(features[0])}")
        output = [elm.tolist() for elm in features]
        return output
