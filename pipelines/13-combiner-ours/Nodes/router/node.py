import logging
import os
import numpy as np
import random


logger = logging.getLogger(__name__)
# NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Node:
    def route(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info("inpu feature type: {}".format(type(features)))
        logging.info(f"features types: {type(features['features'])}")
        logging.info(f"features elements types: {type(features['features'][0])}")
        route = features['route']
        # logging.info(f"model names: {names}")
        # logging.info(f"model meta: {meta}")
        # try:
        # if route == 0:
        #     features['route'] += 1
        # if route != 0:
        #     features['route'] -= 1

        return route

    def init_metadata(self):
        logging.info("metadata method  called")

        meta = {
            "name": "my-model-name",
            "versions": ["my-model-version-01"],
            "platform": "seldon",
            "inputs": [
                {
                    "messagetype": "tensor",
                    "schema": {"names": ["a", "b", "c", "d"], "shape": [4]},
                }
            ],
            "outputs": [{"messagetype": "tensor", "schema": {"shape": [1]}}],
            "custom": {
                "author": "seldon-dev"
            }
        }

        return meta

    # def init_metadata(self):
    #     logging.info('Metadata called!')
    #     meta = {
    #         "name": "router"
    #     }
    #     return meta