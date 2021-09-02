"""interface of the machine learning based
vertical pod autoscalers
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class NonMLInterface(ABC):
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def update(self, observation: np.array):
        """update the resource usage
        """
        pass

    @abstractmethod
    def reset(self):
        """reset the content of the object
        """
        pass

    @abstractmethod
    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
        """
        pass

    @abstractmethod
    def _check_config(self):
        """check the config structure according to
        the recommender method
        """
        pass
