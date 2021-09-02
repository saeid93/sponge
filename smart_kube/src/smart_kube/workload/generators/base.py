from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, config, timesteps):
        self.config = config
        self.timesteps = timesteps
        self._check_config()

    @abstractmethod
    def make_workload(self):
        """generates the workload based
        on each method's different algorithm
        """
        pass

    @abstractmethod
    def _check_config(self):
        """checks if the config of the worklaod is in
        in the correct format
        """
        pass
