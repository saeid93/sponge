import os
import sys
import click
import json
import pickle
import numpy as np

import gym

import smart_kube # noqa
from smart_kube.recommender import (
    Threshold,
    Random,
    Builtin,
    RL,
    LSTM
)
from smart_kube.util import logger

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    CONFIGS_PATH,
    WORKLOADS_PATH
)


def check_env(env, recommender):
    done = False
    _ = env.reset()
    _ = recommender.reset()
    while not done:
        timestamp = env.wall_time
        observation = env.observation
        recommender.update(observation=observation, timestamp=timestamp)
        action = recommender.recommender()
        observation, reward, done, info = env.step(action)
        log_action = {
            "memory": action[[0, 2, 4]].tolist(),
            "cpu": action[[1, 3, 5]].tolist()
        }
        logger.info(log_action)
        env.render()


@click.command()
@click.option('--type-env', required=True,
              type=click.Choice(['sim', 'kube']),
              default='sim')
@click.option('--type-recommender', required=True,
              type=click.Choice(['builtin', 'random', 'threshold',
                                 'lstm', 'rl']),
              default='builtin')
@click.option('--container-id', required=True, type=int, default=0)
@click.option('--workload-id', required=True, type=int, default=12)
@click.option('--seed', required=True, type=int, default=100)
@click.option('--round-robin', required=True, type=bool, default=False)
def main(
    type_env: str,
    type_recommender: str,
    container_id: int,
    workload_id: int,
    seed: int,
    round_robin: bool
        ):
    """
    """
    # -------------- load container config and workload --------------
    # load container config
    # container initial requests and limits
    config: dict = {}
    workload: np.array = np.array([])
    time: np.array = np.array([])
    recommender_file_path = os.path.join(
        CONFIGS_PATH, 'containers', f"{container_id}.json")
    try:
        with open(recommender_file_path) as cf:
            config = json.loads(cf.read())
    except FileNotFoundError:
        print(f"container {container_id} does not exist")

    # load the workoad
    workload_file_path = os.path.join(
        WORKLOADS_PATH, str(workload_id), 'workload.pickle')
    try:
        with open(workload_file_path, 'rb') as in_pickle:
            workload = pickle.load(in_pickle)
    except FileNotFoundError:
        raise Exception(f"workload {workload_id} does not exists")

    # load the time array of the workload
    time_file_path = os.path.join(
        WORKLOADS_PATH, str(workload_id), 'time.pickle')
    try:
        with open(time_file_path, 'rb') as in_pickle:
            time = pickle.load(in_pickle)
    except FileNotFoundError:
        raise Exception(f"workload {workload_id} does not have time array")

    # -------------- make the environment --------------
    # update the passed config to the environment
    config.update({
        'workload': workload,
        'seed': seed,
        'round-robin': round_robin,
        'time': time})

    # picking the right environment
    type_env = {
        'sim': 'SimEnv-v0',
        'kube': 'KubeEnv-v0'
    }[type_env]
    env = gym.make(type_env, config=config)

    # -------------- make the recommender --------------
    # load recommender config
    recommender_file_path = os.path.join(
        CONFIGS_PATH, 'recommender', f"{type_recommender}.json")
    try:
        with open(recommender_file_path) as cf:
            config = json.loads(cf.read())
    except FileNotFoundError:
        print(f"recommender {type_recommender} does not exist")

    # add the action space of the environment to the config
    config.update({'action_space': env.action_space})

    # passing the approperiate recommender
    recommender = {
        'threshold': Threshold,
        'random': Random,
        'builtin': Builtin,
        'rl': RL,
        'lstm': LSTM
        }[type_recommender](config=config)

    # if it's an ML method then load the saved model
    base_class = repr(recommender.__class__.__bases__)
    if 'MLInterface' in base_class:
        # TODO load a saved model
        # TODO recommender.load(model=model)
        pass

    # -------------- run the environment --------------
    check_env(
        env=env,
        recommender=recommender)


if __name__ == "__main__":
    main()
