import logging
import random
import os
from numpy import indices

from torch import threshold


# NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Router:
    def __init__(self) -> None:
        super().__init__()
        logging.info("Router initialized!")
        # self.threshold = float(os.environ["THRESHOLD"])

    def route(self, features, names=[], meta=[]):
        try:
            logging.info(f"Incoming features:\n{features.keys()}\nwas recieved!")
            for k, v in features.items():
                logging.info(f"type {k}: {type(v)}")
            logging.info(f"max_prob_percentage: {features['max_prob_percentage']}")
            logging.info(f"route: {features['route']}")
        except Exception as e:
            logging.error(f"Error: {e}")
        route = features["route"]
        # logging.info("features: %s", features)
        # logging.info("features types %s", type(features))
        # logging.info("names: %s", names)
        # logging.info("meta: %s", meta)
        # indices = features["indices"]
        # percentages = features["percentages"]
        # max_prob_percentage = features["max_prob_percentage"]
        # logging.info(f"model indices: {indices}")
        # # logging.info(f"The threshold is: {self.threshold}")
        # logging.info(f"model percentages: {percentages}")
        # logging.info(f"model max_prob_percentage: {max_prob_percentage}")
        # logging.info(f"threshold: {self.threshold}")
        # # logging.info(f"model meta: {meta}")
        # # route = random.randint(0, NUMBER_OF_ROUTES)
        # logging.info(f"routing to: {0}")
        return route