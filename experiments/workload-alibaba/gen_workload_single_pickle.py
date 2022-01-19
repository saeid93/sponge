"""preprocess data for each container to the
   format of the simulators and predictors
   and save the output to a single pickle file
"""

import pickle
import numpy as np
import os
import sys


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

# format of the output
# dataset = {
#     'namespace_name': {
#         'pod_name': {
#             'container_name': {
#                 'timestamp': (num_timestamp,),
#                 'workload': (2, num_timestamp),
#                 'limit': (2,),
#                 'request': (2,)
#                 }
#             }
#         }
#     }


# input units
# Memory in bytes
# cpu in cores


# output units
# Memory in Megabytes
# cpu in Millicores

cluster_name = "engine-top-ten"
dataset_path = os.path.join(
    ARABESQUE_PATH,
    'etl-outputs',
    f"{cluster_name}.pickle")

with open(dataset_path, 'rb') as in_file:
    dataset = pickle.load(in_file)


output_path = os.path.join(
    WORKLOADS_PATH, 'arabesque-single-file',
    f"{cluster_name}.pickle")


cluster = {}
for namespace, pods in dataset.items():
    cluster[namespace] = {}
    total_pod_count = len(pods)
    pod_number = 0
    for pod_name, container in pods.items():
        cluster[namespace][pod_name] = {}
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
        request_mem = np.nan_to_num(container['request'][0])
        request_mem /= 1e6
        request_mem = request_mem.astype(int)
        request_mem = request_mem.max()
        limit_mem = np.nan_to_num(container['limit'][0])
        limit_mem /= 1e6
        limit_mem = limit_mem.astype(int)
        limit_mem = limit_mem.max()
        # --------- fetch requests and limit cpu ---------
        request_cpu = np.nan_to_num(container['request'][1]).max()
        request_cpu *= 1000
        limit_cpu = np.nan_to_num(container['limit'][1]).max()
        limit_cpu *= 1000
        # --------- info ---------
        cluster[namespace][pod_name].update(
            {
                'workload': workload,
                'time': timestamps,
                'requests': {
                    'memory': float(request_mem),
                    'cpu': request_cpu
                },
                'limits': {
                    'memory': float(limit_mem),
                    'cpu': limit_cpu
                }
            }
        )
        print(f"pod {pod_number} created out of {total_pod_count}")
        pod_number += 1

with open(output_path, 'wb') as out_pickle:
    pickle.dump(cluster, out_pickle)
