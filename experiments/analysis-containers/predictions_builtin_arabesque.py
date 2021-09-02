import os
import sys
import pickle
import numpy as np
import json
from smart_kube.recommender_initial import Builtin
import matplotlib.pyplot as plt
from smart_kube.util import (
    logger,
    plot_recommender,
    plot_histogram
    )

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH
)

# workload
cluster = "portfolio-top-ten"
namespace = "qryfolio-daily"
pod = "qryfolio-cli-backtest-global-q9m8m-1716119528"

# histograms
cpu_first_bucket_size = 0.01
cpu_max_value = 1000
memory_first_bucket_size = 1e7
memory_max_value = 1e12
margin = True
confidence = False
min_resource = False

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

recommendations_memory = []
recommendations_cpu = []

recommender = Builtin(
    cpu_first_bucket_size=cpu_first_bucket_size,
    cpu_max_value=cpu_max_value,
    memory_first_bucket_size=memory_first_bucket_size,
    memory_max_value=memory_max_value,
    margin=margin,
    confidence=confidence,
    min_resource=min_resource)

for i in range(0, workload.shape[1]):
    recommender.update(memory_usage=workload[0, i],
                       cpu_usage=workload[1, i],
                       timestamp=time[i])
    recommendation = recommender.recommender()
    recommendation_formatted = {
        "memory": recommendation[[0, 2, 4]].tolist(),
        "cpu": recommendation[[1, 3, 5]].tolist()
    }
    recommendations_memory.append(recommendation[[0, 2, 4]].tolist())
    recommendations_cpu.append(recommendation[[1, 3, 5]].tolist())
    logger.info(recommendation_formatted)

recommendations_memory = np.array(recommendations_memory).T
recommendations_cpu = np.array(recommendations_cpu).T

# plot recommmender
fig_1 = plot_recommender(
    timestamps=time,
    workload=workload,
    recommendations_memory=recommendations_memory,
    recommendations_cpu=recommendations_cpu)
fig_1.savefig('pic_recommender.png')
plt.close()

# plot histogram
# fig_2 = plot_histogram(
#     memory_histogram=recommender.memory_histogram,
#     histogram=recommender.cpu_histogram,
#     last_bucket_memory=10,
#     last_bucket=39,
#     width=0.01,
#     width_memory=0.01
# )
fig_2.savefig('pic_histogram.png')
plt.close()
