from .ml_interface import MLInterface
from typing import Dict, Any


class RL(MLInterface):
    def __init__(self, config: Dict[str, Any]):
        pass

    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
        """
        raise NotImplementedError

    def fit(self):
        """training function of the selected ml model
        """
        # TODO load approperiate loss function
        raise NotImplementedError

    def predict(self):
        """ml model predictor
        used in the recommender function
        """
        raise NotImplementedError

    def load(self, model):
        """load a saved ml model
        """
        self.model = model
        raise NotImplementedError

    def _check_config(self):
        """check the config structure according to
        the recommender method
        """
        pass
