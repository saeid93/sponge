import logging
import os
import numpy as np
import random


logger = logging.getLogger(__name__)
# NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Node:
    def route(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        route = int(np.random.choice([-2, 0]))
        # route = random.randint(-2, 0)
        # logging.info(f"model route: {route}")
        # route = 0
        # if features[0][0] == 1:
        #     route = 1
        #     logging.info(f"model route: {route}")
        # else:
        #     logging.info(f"routing to: {route}")

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