from tqdm import tqdm
import numpy as np
from .base import Base


class Step(Base):
    def __init__(self, config: dict, timesteps: int):
        super().__init__(config, timesteps)
        self.num_resources = 2
        workloads_var = config['workload_var']
        start_workload = config['start_workload']
        self.cpu_max_usage = config['max_usage']['cpu']
        self.ram_max_usage = config['max_usage']['memory']

        self.start_workload = \
            np.transpose(np.array(start_workload))
        self.workloads_steps_units = \
            np.transpose(np.array(workloads_var['steps_unit']))
        self.workloads_max_steps = \
            np.transpose(np.array(workloads_var['max_steps']))

    def make_workload(self):
        """
                        different types
            memory    |                |
            cpu       |                |

        """

        workload = np.zeros((self.num_resources,
                             self.timesteps))
        workload[:, 0] = self.start_workload

        # generate workloads based-on fraction of usage
        for col in tqdm(range(1, self.timesteps)):
            num_steps = np.random.randint(-self.workloads_max_steps,
                                          self.workloads_max_steps+1)
            steps = num_steps * self.workloads_steps_units
            workload[:, col] = workload[:, col-1] + steps
            workload[workload < 0] = 0
            workload[workload > 1] = 1

        # make the fraction into actual resource usage
        workload[0, ] *= self.ram_max_usage
        workload[1, ] *= self.cpu_max_usage
        return workload

    def _check_config(self):
        return super()._check_config()
        # TODO implement
