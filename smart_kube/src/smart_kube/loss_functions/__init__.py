"""interface of the machine learning based
vertical pod autoscalers
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, input: np.ndarray):
        """compute the loss function result
        """
        pass
