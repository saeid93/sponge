import numpy as np
import random
from smart_kube.util import plot_workload
from smart_kube.workload.generators import (
    Alibaba,
    Arabesque,
    Azure,
    Sinusoidal,
    LowHigh,
    Step,
    Constant
)


class SyntheticWorkloadGenerator:
    def __init__(self,
                 workload_type,
                 #  plot_smoothing,
                 time_interval,
                 seed,
                 timesteps,
                 container,
                 config):
        """
            workload generator
        """
        self.time_interval = time_interval
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)
        self.timesteps = timesteps
        self.request_cpu = container['requests']['cpu']
        self.limit_cpu = container['limits']['cpu']
        self.request_mem = container['requests']['memory']
        self.limit_mem = container['limits']['memory']
        assert workload_type in [
            'alibaba',
            'arabesque',
            'azure',
            'sinusoidal',
            'lowhigh',
            'step',
            'constant'
        ], f"Unknown workload type {workload_type}"
        generators = {
            'alibaba': Alibaba,
            'arabesque': Arabesque,
            'azure': Azure,
            'sinusoidal': Sinusoidal,
            'step': Step,
            'lowhigh': LowHigh,
            'constant': Constant
        }
        self.generator = generators[workload_type](
            config,
            timesteps
        )
        # self.plot_smoothing = plot_smoothing

    def make_workload(self):
        """

        start workload:
        resource usage is described in fractions [0, 1]
        e.g. 0.5 shows 50 percent of the ram is occupied

                           different types
            memory       |                |
            cpu          |                |

        use the approperiate type for generating the workloads
        """
        workload = self.generator.make_workload()
        workload = np.round(workload).astype(int)
        time = self.make_time()
        fig = plot_workload(
            time,
            workload,
            self.request_cpu,
            self.limit_cpu,
            self.request_mem,
            self.limit_mem)
        return workload, fig, time

    def make_time(self):
        time = np.arange(self.timesteps) * self.time_interval
        return time
