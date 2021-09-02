from .nonml_interface import NonMLInterface
import numpy as np
from typing import Dict, Any


class Threshold(NonMLInterface):
    def __init__(self, config: Dict[str, Any]):
        self._check_config(config)
        self.target_cpu = config['target']['cpu']
        self.target_memory = config['target']['memory']
        self.upper_bound_cpu = config['upper_bound']['cpu']
        self.upper_bound_memory = config['upper_bound']['memory']
        self.lower_bound_cpu = config['lower_bound']['cpu']
        self.lower_bound_memory = config['lower_bound']['memory']

    def update(self, observation: np.array):
        pass

    def reset(self):
        pass

    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
         ram_lower_bound   cpu_lower_bound
        [                |                |

         ram_target   cpu_target
        |           |            |

         ram_higher_bound   cpu_higher_bound
        |                 |                 ]
        """
        recommendation = np.array([
            self.lower_bound_memory,
            self.lower_bound_cpu,
            self.target_memory,
            self.target_cpu,
            self.upper_bound_memory,
            self.upper_bound_cpu
        ])
        return recommendation

    def _check_config(self, config):
        pass
