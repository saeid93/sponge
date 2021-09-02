import gym
from gym.utils import seeding
from typing import ( # noqa
    List,
    Dict,
    Any
)


class KubeEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.seed: int = config['seed']
        # np.random.seed(self.seed)
        self.np_random = seeding.np_random(self.seed)

        self._check_config(config)

    # add async function to get the workload periodically
    def _check_config(config):
        pass
