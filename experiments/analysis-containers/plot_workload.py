import os
import sys
import pickle
import numpy as np
import json
from smart_kube.recommender_initial import Builtin
from smart_kube.util import (
    logger,
    plot_workload
    )

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH
)

# workload
cluster = "portfolio-july-all"
namespace = "qryfolio"
pod = "optfolio-8mzzw-796513259"


# -------------- load the workload --------------
config: dict = {}
workload: np.array = np.array([])
time: np.array = np.array([])
pod_path = os.path.join(
    WORKLOADS_PATH, 'arabesque', cluster, namespace, pod)

# load container config
# container initial requests and limits
container_file_path = os.path.join(pod_path, "container.json")
try:
    with open(container_file_path) as cf:
        config = json.loads(cf.read())
except FileNotFoundError:
    print(f"pod {pod} does not have a container")

# load the workoad
workload_file_path = os.path.join(pod_path, 'workload.pickle')
try:
    with open(workload_file_path, 'rb') as in_pickle:
        workload = pickle.load(in_pickle)
except FileNotFoundError:
    raise Exception(f"pod {pod} does not exists")

# load the time array of the workload
time_file_path = os.path.join(pod_path, 'time.pickle')
try:
    with open(time_file_path, 'rb') as in_pickle:
        time = pickle.load(in_pickle)
except FileNotFoundError:
    raise Exception(f"pod {pod} does not have time array")

fig = plot_workload(
    timestamps=time,
    workload=workload,
    request_memory=np.max(workload[0])*1.15,
    limit_memory=np.max(workload[0])*1.15,
    request_cpu=np.max(workload[1])*1.15,
    limit_cpu=np.max(workload[1])*1.15)
    # request_memory=config['requests']['memory'],
    # limit_memory=config['limits']['memory'],
    # request_cpu=config['requests']['cpu'],
    # limit_cpu=config['limits']['cpu'])


fig.savefig('pic_workload_max_pred.png')
