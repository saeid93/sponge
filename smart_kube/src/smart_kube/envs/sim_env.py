import numpy as np
from typing import (
    List,
    Dict,
    Any
)
import pprint
import yaml

import gym
from gym.utils import seeding
from gym.spaces import Box
from colorama import (
    Fore,
    Style
)

from smart_kube.util import logger
from smart_kube.util.constants import LIMIT_RANGE

pp = pprint.PrettyPrinter()


class RecommenderSpace(Box):
    """The space class that is for generating values that are
    lower_bound<target<upperbound
    """
    def sample(self):
        lower_bound = np.random.randint(
            low=self.low[0:2],
            high=self.high[0:2],
            size=2
        )
        target = np.random.randint(
            low=lower_bound,
            high=self.high[0:2],
            size=2
        )
        upper_bound = np.random.randint(
            low=target,
            high=self.high[0:2],
            size=2
        )
        samp = np.concatenate((
            lower_bound,
            target,
            upper_bound
        ))
        return samp

    def contains(self, x):
        if not super().contains(x):
            return False
        if np.alltrue(x[0:2] <= x[2:4]) and np.alltrue(x[2:4] <= x[4:6]):
            return True
        return False


class SimEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        """reads the initial configuration from the config file

        Args:
            config (Dict[str, Any]): [description]
        selfs:
            requests:
                ram cpu
                [  |   ]
            limits:
                ram cpu
                [  |   ]

        """
        # check if the config is in the right format
        super().__init__()
        # self._check_config(config)

        # set up the seeds for reproducable resutls
        self.seed: int = config['seed']
        np.random.seed(self.seed)
        self.np_random = seeding.np_random(self.seed)

        # time variable (for computing clock time in simulation
        # and reading metrics in the emulations)
        self.time = config['time']

        # container name
        self.container_name: str = config['container_name']

        # initail resource requests
        self.initial_requests = np.zeros(2)
        self.initial_requests[0] = config['requests']['memory']
        self.initial_requests[1] = config['requests']['cpu']
        self.requests = self.initial_requests.copy()

        # initial resource limits
        self.initial_limits = np.zeros(2)
        self.initial_limits[0] = config['limits']['memory']
        self.initial_limits[1] = config['limits']['cpu']
        self.limits = self.initial_limits.copy()

        # limit ranges
        self.limit_range_min = np.zeros(2)
        self.limit_range_max = np.zeros(2)
        self.max_limit_request_ratio = np.zeros(2)
        self.limit_range_max[0] = LIMIT_RANGE['max']['memory']
        self.limit_range_max[1] = LIMIT_RANGE['max']['cpu']
        self.limit_range_min[0] = LIMIT_RANGE['min']['memory']
        self.limit_range_min[1] = LIMIT_RANGE['min']['cpu']
        self.max_limit_request_ratio[0] = LIMIT_RANGE[
            'max_limit_request_ratio']['memory']
        self.max_limit_request_ratio[1] = LIMIT_RANGE[
            'max_limit_request_ratio']['cpu']

        # whether we want to end at the end of the workload
        # or start over from the begining
        self.round_robin: bool = config['round-robin']

        # ratio of the request to limit
        # (constant and stays the same during the experimetns)
        self.limit_request_ratio =\
            self.initial_limits/self.initial_requests

        # kubernetes value checks
        self._kubernetes_checks()

        # initiate the observation and action space
        self.observation_space, self.action_space =\
            self._setup_space()

        # workload of resources usage
        # resource usage        timestep
        # ram (in megabayes) |    ...     |
        # cpu (in milicores) |    ...     |
        self.workload: np.array = config['workload']
        self.total_timesteps = self.workload.shape[1]
        self.timestep = 0
        self.global_timestep = 0
        self.prev_observation = self.observation.copy()
        self.recreation_flag = False
        _ = self.reset()

        logger.info("container initialised!")
        initial_observation = {
            'container_name': self.container_name,
            'requests': {
                'memory': f"{self.requests[0]}Mi",
                'cpu': f"{self.requests[1]}m"
            },
            'limits': {
                'memory': f"{self.limits[0]}Mi",
                'cpu': f"{self.limits[1]}m"
            }
        }
        logger.info(yaml.dump(initial_observation,
                              default_flow_style=False))

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        """
        self.timestep = 0
        self.global_timestep = 0
        self.recreation_flag = False
        self.prev_observation = self.observation.copy()
        self.requests = self.initial_requests.copy()
        self.limits = self.initial_limits.copy()
        return self.observation

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        """
        # assert self.action_space.contains(action),\
        #     f"action {action} out of action space {self.action_space}"
        self.prev_observation = self.observation.copy()
        self.action = action
        reward = self._calc_reward()
        # recreate the pod if needed
        # (equivalent to taking a step in rl terminology)
        if self._recreation_needed:
            self.recreation_flag = True
            self._recreate()
        self.global_timestep += 1
        if self.round_robin:
            self.timestep = self.global_timestep % self.total_timesteps
        else:
            self.timestep = self.global_timestep

        return self.observation, reward, self.done, self.info

    def render(self, mode='human'):
        """Renders the environment.
        """
        print('\n')
        print(30*'=', '\n',
              f"workload timestep: {self.timestep}\n",
              f"total timesteps: {self.global_timestep}\n",
              30*'=', '\n')
        print(15*'-', " action ", 15*'-')
        print(yaml.dump(self.action_formatted,
                        default_flow_style=False))
        if self.recreation_flag:
            print(Fore.RED, 'container was recreated!')
            print('\n')
            print(15*'-', " prev observation ", 15*'-')
            print(yaml.dump(self.prev_observation_formatted,
                            default_flow_style=False))
            self.recreation_flag = False
        print(Style.RESET_ALL)
        print(15*'-', " observation in the current timestep ", 15*'-')
        print(yaml.dump(self.observation_formatted,
                        default_flow_style=False))

    def _setup_space(self):
        """Make the observation and action space
        observation space:

         ram_usage cpu_usage ram_request cpu_request
        [         |         |           |           ]

        action space:

         ram_lower_bound   cpu_lower_bound
        [                |                |

         ram_target   cpu_target
        |           |            |

         ram_higher_bound   cpu_higher_bound
        |                 |                 ]

        Units:

        cpu units: milicores
        memory units: megabytes
        """
        MIN = 0
        MAX = 100000

        # observation space
        obs_lower_bound = np.concatenate((
            np.array([MIN, MIN]),
            self.limit_range_min))
        obs_upper_bound = np.concatenate((
            np.array([MAX, MAX]),
            self.limit_range_max))
        observation_space = Box(
            low=obs_lower_bound, high=obs_upper_bound,
            shape=(4,), dtype=float)

        # action space
        act_lower_bound = np.array(
            self.limit_range_min.tolist() * 3
            )
        act_upper_bound = np.array(
            self.limit_range_max.tolist() * 3
            )
        action_space = RecommenderSpace(
            low=act_lower_bound, high=act_upper_bound,
            shape=(6,), dtype=float)
        return observation_space, action_space

    # def _check_config(self, config):
    #     """check if the config is in the correct format
    #     """
    #     allowed_items = [
    #         'container_name', 'resources',
    #         'requests', 'limits', 'workload', 'time',
    #         'seed', 'round-robin'
    #     ]
    #     for key, _ in config.items():
    #         assert key in allowed_items, (
    #             f"<{key}> is not an allowed items for"
    #             " the environment config")

    #     def _type_check(
    #         obs: List, type_of: type, type_name: str
    #             ):
    #         for item in obs:
    #             assert type(config[item]) == type_of,\
    #                 f"<{item}> must be a {type_name}"

    #     _type_check(['container_name'], str, 'string')
    #     _type_check(['workload', 'time'], np.ndarray, 'numpy array')
    #     _type_check(['seed'], int, 'integer')
    #     _type_check(['round-robin'], bool, 'boolean')

    #     # check for the request and limits contents
    #     constriants = ['requests', 'limits']
    #     resources = ['memory', 'cpu']
    #     for constraint in constriants:
    #         for key, value in config[constraint].items():
    #             assert key in resources, f"Unknown resource <{key}>"
    #             assert type(value) == int, (
    #                 f"<{key}> value must be and integer,"
    #                 f" got a <{type(value)}> instead")

    def _kubernetes_checks(self):
        """Kubernetes value checks
        """
        # limits should not be greater than requests
        assert np.alltrue(self.initial_requests <= self.initial_limits),\
            (f"limits values <{self.initial_limits}> must be smaller than"
             f" requests values <{self.initial_requests}>")

        # check limit ranges logic
        assert np.alltrue(self.limit_range_min <= self.limit_range_max),\
            (f"min limit range values {self.limit_range_min} must be smaller"
             f" than max limit range values {self.limit_range_max}")
        assert np.alltrue(self.max_limit_request_ratio >= 1),\
            (f"max_limit_request_ratio <{self.max_limit_request_ratio}>"
             " must be greater than one")

        # check initial request and limits against the limit ranges
        assert np.alltrue(self.initial_requests >= self.limit_range_min),\
            (f"initial requests  <{self.initial_requests}> must be "
             f" greater than the min limit range <{self.limit_range_min}>")
        assert np.alltrue(self.initial_requests <= self.limit_range_max),\
            (f"initial requests <{self.initial_requests}> must be smaller"
             f" than the max limit range values {self.limit_range_max}")

        # check the limit to range ratio
        assert np.alltrue(
            self.limit_request_ratio <= self.max_limit_request_ratio),\
            ("initial request to limit request ratio "
             f"<{self.limit_request_ratio}>"
             f"greater than max request to limit ratio "
             f"<{self.max_limit_request_ratio}>")

    def _calc_reward(self):
        """Calculate some rewards based-on the slack
        and other metrics
        """
        # TODO should come from the cost function
        # and the histogram
        reward = 1
        return reward

    def _recreate(self):
        """recreate the pod based-on new criteria
        recreation means changing the requests and the limits
        """
        # TODO check limit_range criteria with gym spaces functionalities
        self.prev_observation = self.observation.copy()
        self.requests = self.target.copy()
        self.limits = (
            self.target.copy() * self.limit_request_ratio).astype(int)

    @property
    def time_history(self):
        """get the resource usage from the past up to the current
        point of time in reverse order
        e.g. to get the most recent resource usage of the previous 5 step
        """
        return self.time[0:self.timestep+1]

    @property
    def resource_usage_history(self):
        """get the resource usage from the past up to the current
        point of time in reverse order
        e.g. to get the most recent resource usage of the previous 5 step
        """
        return self.workload[:, 0:self.timestep+1]

    @property
    def observation(self):
        self.resource_usage_current
        obs = np.concatenate((
            self.resource_usage_current,
            self.requests)
        )
        return obs

    @property
    def observation_formatted(self):
        """actions in format consistent with the original autosacler format
           e.g. {'usage': {'cpu': '1m', 'memory': '524Mi'},
                 'requests': {'cpu': '1m', 'memory': '1000Mi'},
                 'limits': {'cpu': '10m', 'memory': '2000Mi'}}
        """
        observation_formatted = {
            'usage': {
                'memory': f"{self.resource_usage_current[0]}Mi",
                'cpu': f"{self.resource_usage_current[1]}m"
            },
            'requests': {
                'memory': f"{self.requests[0]}Mi",
                'cpu': f"{self.requests[1]}m"
            },
            'limits': {
                'memory': f"{self.limits[0]}Mi",
                'cpu': f"{self.limits[1]}m"
            }
        }
        return observation_formatted

    @property
    def prev_observation_formatted(self):
        """actions in format consistent with the original autosacler format
           e.g. {'usage': {'cpu': '1m', 'memory': '524Mi'},
                 'requests': {'cpu': '1m', 'memory': '1000Mi'},
                 'limits': {'cpu': '10m', 'memory': '2000Mi'}}
        """
        memory_limit = (
            self.prev_observation[2] * self.limit_request_ratio[0]).astype(int)
        cpu_limit = (
            self.prev_observation[3] * self.limit_request_ratio[1]).astype(int)
        prev_observation_formatted = {
            'usage': {
                'memory': f"{self.prev_observation[0]}Mi",
                'cpu': f"{self.prev_observation[1]}m"
            },
            'requests': {
                'memory': f"{self.prev_observation[2]}Mi",
                'cpu': f"{self.prev_observation[3]}m"
            },
            'limits': {
                'memory': f"{memory_limit}Mi",
                'cpu': f"{cpu_limit}m"
            }
        }
        return prev_observation_formatted

    @property
    def action_formatted(self):
        """actions in format consistent with the original autosacler format
           e.g. {'containerName': 'experimental-deployment',
                'lowerBound': {'cpu': '1m', 'memory': '524Mi'},
                'target': {'cpu': '1m', 'memory': '734Mi'},
                'upperBound': {'cpu': '210m', 'memory': '1369Mi'}}
        """
        action_formatted = {
           'containerName': self.container_name,
           'lowerBound': {
               'memory': f"{self.lower_bound[0]}Mi",
               'cpu': f"{self.lower_bound[1]}m"
           },
           'target': {
               'memory': f"{self.target[0]}Mi",
               'cpu': f"{self.target[1]}m"
           },
           'upperBound': {
               'memory': f"{self.upper_bound[0]}Mi",
               'cpu': f"{self.upper_bound[1]}m"
           }
        }
        return action_formatted

    @property
    def resource_usage_current(self):
        """Container's resource usage at the current timestep
             ram cpu
            |       |
        """
        return np.round(self.workload[:, self.timestep])

    @property
    def wall_time(self):
        return self.time[self.timestep]

    @property
    def lower_bound(self):
        """descriptive recommended lower bound value from the recommender

        Returns:
            np.array: lower bound ram and cpu
        """
        return self.action[0:2]

    @property
    def target(self):
        """descriptive target value from the recommender

        Returns:
            np.array: target for ram and cpu
        """
        return self.action[2:4]

    @property
    def upper_bound(self):
        """descriptive upper bound from the recommender

        Returns:
            np.array: lower bound ram and cpu
        """
        return self.action[4:6]

    @property
    def _recreation_needed(self):
        """check the recreation conditions
        """
        if not np.alltrue(self.lower_bound < self.resource_usage_current):
            return True
        if not np.alltrue(self.resource_usage_current < self.upper_bound):
            return True
        if not np.alltrue(self.resource_usage_current < self.limits):
            return True
        return False

    @property
    def done(self):
        """check if the agent is at a finish state or not
        """
        done = False
        if not self.round_robin:
            if self.timestep == self.total_timesteps-1:
                done = True
        return done

    @property
    def info(self):
        """info dictionary for the step function
        """
        return self.observation_formatted
