import os
import sys
import pickle
import numpy as np
from smart_kube.recommender_initial import Builtin
from smart_kube.util import (
    logger,
    plot_recommender
    )

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH
)

# workload information
workload_id = 1

# histograms
cpu_first_bucket_size = 0.01
cpu_max_value = 1000
memory_first_bucket_size = 1e7
memory_max_value = 1e12
margin = True
confidence = False
min_resource = False

# -------------- load the workload --------------

workload_path = os.path.join(
    WORKLOADS_PATH, 'synthetic', str(workload_id))

# load the workoad
workload_file_path = os.path.join(workload_path, 'workload.pickle')
try:
    with open(workload_file_path, 'rb') as in_pickle:
        workload = pickle.load(in_pickle)
except FileNotFoundError:
    raise Exception(f"workload {workload_id} does not exists")

# load the time array of the workload
time_file_path = os.path.join(workload_path, 'time.pickle')
try:
    with open(time_file_path, 'rb') as in_pickle:
        time = pickle.load(in_pickle)
except FileNotFoundError:
    raise Exception(f"workload {workload_id} does not have time array")


memory_recommendations = []
cpu_recommendations = []

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
    memory_recommendations.append(recommendation[[0, 2, 4]].tolist())
    cpu_recommendations.append(recommendation[[1, 3, 5]].tolist())
    logger.info(recommendation_formatted)

memory_recommendations = np.array(memory_recommendations).T
cpu_recommendations = np.array(cpu_recommendations).T

fig = plot_recommender(
    timestamps=time,
    workload=workload,
    recommendations_memory=memory_recommendations,
    recommendations_cpu=cpu_recommendations)

fig.savefig('pic_recommender.png')
