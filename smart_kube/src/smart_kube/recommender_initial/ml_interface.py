"""interface of the machine learning based
vertical pod autoscalers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union

from smart_kube.envs import SimEnv, KubeEnv
# TODO add loss function loader here


class MLInterface(ABC):
    def __init__(self, config: Dict[str, Any], env: Union[SimEnv, KubeEnv]):
        pass

    @abstractmethod
    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
        """
        pass

    @abstractmethod
    def fit(self):
        """training function of the selected ml model
        """
        # TODO load approperiate loss function
        pass

    @abstractmethod
    def predict(self):
        """ml model predictor
        used in the recommender function
        """
        pass

    @abstractmethod
    def load(self, data):
        """load a saved ml model
        """
        pass

    @abstractmethod
    def _check_config(self):
        """check the config structure according to
        the recommender method
        """
        pass
