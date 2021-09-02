from .base import Base
import numpy as np


class Sinusoidal(Base):
    def __init__(self, config, timesteps):
        super().__init__(config, timesteps)
        self.num_resources = 2
        self.cpu_max_usage = config['max_usage']['cpu']
        self.ram_max_usage = config['max_usage']['memory']
        self.num_peaks = config['num_peaks']

    def make_workload(self):
        """
                        different types
            memory    |                |
            cpu       |                |

        """

        workload = np.zeros((self.num_resources,
                             self.timesteps))
        timesteps = np.linspace(
            0, self.num_peaks * np.pi, self.timesteps)
        amplitude = np.sin(timesteps)
        amplitude = (amplitude+1)/2
        workload[0] = amplitude * self.ram_max_usage
        workload[1] = amplitude * self.cpu_max_usage
        return workload

    def _check_config(self):
        return super()._check_config()
        # TODO implement
