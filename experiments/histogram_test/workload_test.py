import numpy as np
import os
import sys
import pickle

from smart_kube.util import Histogram

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH
)


cpu_first_bucket_size = 0.01
cpu_max_value = 1000

memory_first_bucket_size = 1e7
memory_max_value = 1e12

# -------------- load the workoad --------------
workload_id = 13
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
memory_workload = workload[0] * 10e6
cpu_workload = workload[1] / 1000


# -------------- load the workoad --------------

timesteps = 10
time_interval = 60

time = np.arange(timesteps) * time_interval
memory_workload = np.array([1e9, 2e9, 3e9, 1e11, 33445333333])
cpu_workload = np.array([1, 2, 3, 100, 1000, 0.1, 0.3])


# ------------- memory -------------
memory_histogram = Histogram(
    max_value=memory_max_value,
    first_bucket_size=memory_first_bucket_size,
    time_decay=False
)
for value in memory_workload:
    memory_histogram.add_sample(value=value, weight=1, timestamp=1)

print(f"time factor:\n{memory_histogram.time}\n")  # TODO add after fix decay
print(f"memory histogram:\n{memory_histogram.bucket_weight}\n")

# ------------- cpu -------------
cpu_histogram = Histogram(
    max_value=cpu_max_value,
    first_bucket_size=cpu_first_bucket_size,
    time_decay=False
)
for value in cpu_workload:
    cpu_histogram.add_sample(value=value, weight=1, timestamp=1)

print(f"time factor:\n{cpu_histogram.time}\n")  # TODO add after fix decay
print(f"cpu histogram:\n{cpu_histogram.bucket_weight}\n")
