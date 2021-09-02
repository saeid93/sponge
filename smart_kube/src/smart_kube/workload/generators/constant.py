import numpy as np
from .base import Base


class Constant(Base):
    def __init__(self, config: dict, timesteps: int):
        super().__init__(config, timesteps)
        self.num_resources = 2
        self.cpu_usage = config['usage']['cpu']
        self.memory_usage = config['usage']['memory']
        # TODO check if min is smaller than max

    def make_workload(self):
        """
                        different types
            memory    |                |
            cpu       |                |
        """

        workload = np.ones((self.num_resources,
                            self.timesteps))

        workload[0] *= self.memory_usage
        workload[1] = self.cpu_usage
        return workload

    def _check_config(self):
        return super()._check_config()
        # TODO implement
