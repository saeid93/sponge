import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class Node:
    def aggregate(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        return [x.tolist() for x in features]
