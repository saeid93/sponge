from .nonml_interface import NonMLInterface
from typing import Dict, Any
from smart_kube.envs.sim_env import RecommenderSpace
import numpy as np


class Random(NonMLInterface):
    def __init__(self, config: Dict[str, Any]):
        self._check_config(config)
        self.action_space: RecommenderSpace = config['action_space']

    def update(self, observation: np.array):
        pass

    def reset(self):
        pass

    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
        """
        recommendation = self.action_space.sample()
        return recommendation

    def _check_config(self, config):
        pass
