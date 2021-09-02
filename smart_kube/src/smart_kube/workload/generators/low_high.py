import numpy as np
from .base import Base


class LowHigh(Base):
    def __init__(self, config: dict, timesteps: int):
        super().__init__(config, timesteps)
        self.num_resources = 2
        self.cpu_max_usage = config['max_usage']['cpu']
        self.memory_max_usage = config['max_usage']['memory']
        self.cpu_min_usage = config['min_usage']['cpu']
        self.memory_min_usage = config['min_usage']['memory']
        # TODO check if min is smaller than max

    def make_workload(self):
        """

                        different types
            memory    |                |
            cpu       |                |

        """

        workload = np.zeros((self.num_resources,
                             self.timesteps))
        memory_low_high = int(self.timesteps/self.num_resources) * [
            self.memory_min_usage,
            self.memory_max_usage
        ]
        cpu_low_high = int(self.timesteps/self.num_resources) * [
            self.cpu_min_usage,
            self.cpu_max_usage
        ]
        workload[0] = memory_low_high[0:self.timesteps]
        workload[1] = cpu_low_high[0:self.timesteps]
        return workload

    def _check_config(self):
        return super()._check_config()
        # TODO implement
