import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class Node:
    def predict(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        route = int(np.random.choice([-2, 0]))
        features *= 2
        if route == 0:
            return {
                "features": features.tolist(),
                "names": names,
                "route": route,
                "model": 'model1'}
        if route != 0:
            return {
                "features": features.tolist(),
                "names": names,
                "route": route,
                "model": 'model8'}

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
    #         "name": "model-one"
    #     }
    #     return meta
