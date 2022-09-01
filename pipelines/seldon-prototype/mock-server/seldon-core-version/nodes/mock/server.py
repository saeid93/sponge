import logging
import os
import numpy as np
import time

logger = logging.getLogger(__name__)
PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']

class Mock(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        try:
            self.MY_ENV = os.environ['MY_ENV']
            logging.info(f'MY_ENV set to: {self.MY_ENV}')
        except KeyError as e:
            self.MY_ENV = 'MY_ENV' 
            logging.warning(
                f"MY_ENV env variable not set, using default value: {self.MY_ENV}")

    def load(self):
        try:
            logger.info('Loading the ML models')
            self.loaded = True
            self.model = lambda l: 1 # some batching
            logger.info('model loading complete!')
        except OSError:
            pass

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        # logger.info(f"Incoming input:\n{X}\nwas recieved!")
        arrival_time = time.time()
        former_steps_timing = None
        if X is dict and 'time' in X.item():
            former_steps_timing = X['time']
        out = self.model(X)
        serving_time = time.time()
        logger.info(f"Arrival time: {arrival_time}\n!")
        logger.info(f"Serving time: {serving_time}\n!")
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}": arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}": serving_time,
        }
        if former_steps_timing is not None:
            timing.update(former_steps_timing)
        output = {
            "time": timing,
            "output": out
        }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
