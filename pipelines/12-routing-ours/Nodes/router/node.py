import logging
import os
import numpy as np
import random


logger = logging.getLogger(__name__)
NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Node:
    def route(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        route = random.randint(0, NUMBER_OF_ROUTES)
        logging.info(f"routing to: {route}")
        return route