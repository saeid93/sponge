"""preprocess data for each container to the
   format of the simulators and predictors
   and save the output to a separate folder
   per each container
"""

import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import json
import shutil
from smart_kube.util import plot_workload


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH,
    ARABESQUE_PATH,
)

# options: engine-july-all | engine-top-ten | portfolio-july-all
#          portfolio-top-ten

# format of the datasets
# dataset = {
#     'namespace_name': {
#         'pod_name': {
#             'container_name': {
#                 'timestamp' -> (num_timestamp,),
#                                 evictable memory           | non-evictable memory |      cpu
#                 'workload' -> ( np.array (num_timestamp,1)  ,np.array  (num_timestamp,1),np.array  (num_timestamp,1))
#                              memory                      cpu
#                 'limit' -> ( np.array (num_timestamp,1), np.array  (num_timestamp,1))
#                                memory                      cpu
#                 'request' -> ( np.array (num_timestamp,1), np.array  (num_timestamp,1))
#                 }
#             }
#         }
#     }

cluster_name = "engine-top-ten"
dataset_path = os.path.join(
    ARABESQUE_PATH,
    'etl-outputs',
    f"{cluster_name}.pickle")

with open(dataset_path, 'rb') as in_file:
    dataset = pickle.load(in_file)


cluster_folder_path = os.path.join(
    WORKLOADS_PATH, 'arabesque', cluster_name)
if not os.path.exists(cluster_folder_path):
    os.makedirs(cluster_folder_path)


for namespace, pods in dataset.items():
    namespace_path = os.path.join(cluster_folder_path, namespace)
    if not os.path.exists(namespace_path):
        os.makedirs(namespace_path)
    else:
        shutil.rmtree(namespace_path)
        os.makedirs(namespace_path)
    total_pod_count = len(pods)
    pod_number = 0
    for pod_name, container in pods.items():
        pod_path = os.path.join(namespace_path, pod_name)
        if not os.path.exists(pod_path):
            os.mkdir(pod_path)
        # each pod contains a container main - based-on Argos
        container = container['main']
        # --------- get the timestamp ---------
        # we have shifted all the time to zero
        timestamps = container['timestamp']
        # --------- get the workload - memory ---------
        # sum up the evictable and non-evictable memory
        # set nans to zero
        # convert units from bytes to megabytes
        workload_mem = np.nan_to_num(container['workload'][0]) +\
            np.nan_to_num(container['workload'][1])
        workload_mem = workload_mem/1e6
        workload_mem = workload_mem.astype(int)
        workload_mem = np.squeeze(workload_mem)
        # --------- get the workload - cpu ---------
        # set negatives to zero
        # convert units from cores to millicores
        workload_cpu = container['workload'][2]
        workload_cpu[np.where(workload_cpu < 0)[0]] = 0
        workload_cpu *= 1000
        workload_cpu = workload_cpu.astype(int)
        workload_cpu = np.squeeze(workload_cpu)
        # --------- stack mem and cpu workloads ---------
        workload = np.stack((workload_mem, workload_cpu))
        # --------- fetch requests and limit memory ---------
        request_memory = np.nan_to_num(container['request'][0])
        request_memory /= 1e6
        request_memory = request_memory.astype(int)
        request_memory = request_memory.max()
        limit_memory = np.nan_to_num(container['limit'][0])
        limit_memory /= 1e6
        limit_memory = limit_memory.astype(int)
        limit_memory = limit_memory.max()
        # --------- fetch requests and limit cpu ---------
        request_cpu = np.nan_to_num(container['request'][1]).max()
        request_cpu *= 1000
        limit_cpu = np.nan_to_num(container['limit'][1]).max()
        limit_cpu *= 1000
        # --------- info ---------
        info = {
            'container_name': pod_name,
            'requests': {
                'memory': float(request_memory),
                'cpu': request_cpu
            },
            'limits': {
                'memory': float(limit_memory),
                'cpu': limit_cpu
            }
        }
        # save the information and workload in the folder
        with open(os.path.join(pod_path, 'container.json'), 'x') as out_file:
            json.dump(info, out_file, indent=4)
        with open(os.path.join(pod_path,
                  'workload.pickle'), 'wb') as out_pickle:
            pickle.dump(workload, out_pickle)
        with open(os.path.join(pod_path, 'time.pickle'), 'wb') as out_pickle:
            pickle.dump(timestamps, out_pickle)
        fig = plot_workload(
            timestamps=timestamps,
            workload=workload,
            request_cpu=request_cpu,
            limit_cpu=limit_cpu,
            request_memory=request_memory,
            limit_memory=limit_memory)
        fig.savefig(os.path.join(pod_path, 'figure.png'))
        plt.close()
        print(f"pod {pod_number} created out of {total_pod_count}")
        pod_number += 1
