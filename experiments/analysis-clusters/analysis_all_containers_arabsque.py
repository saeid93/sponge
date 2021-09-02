import pickle
import numpy as np
import os
import sys
from smart_kube.recommender_initial import Builtin

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH,
    ANALYSIS_CONTAINERS_PATH
)

# options: engine-july-all | engine-top-ten | portfolio-july-all
#          portfolio-top-ten


cluster_name = "engine-top-ten"

cluster_path = os.path.join(
    WORKLOADS_PATH,
    'arabesque-single-file',
    f"{cluster_name}.pickle")

with open(cluster_path, 'rb') as in_file:
    cluster = pickle.load(in_file)

# histograms
cpu_first_bucket_size = 0.01
cpu_max_value = 1000
memory_first_bucket_size = 1e7
memory_max_value = 1e12
time_decay = True
margin = True
confidence = False
min_resource = False


# per containers stats
# Make two pandas dataframe/dictionary and add all stats


def builtin(workload: np.array, time: np.array):
    recommendations_memory = []
    recommendations_cpu = []

    recommender = Builtin(
        cpu_first_bucket_size=cpu_first_bucket_size,
        cpu_max_value=cpu_max_value,
        memory_first_bucket_size=memory_first_bucket_size,
        memory_max_value=memory_max_value,
        margin=margin,
        confidence=confidence,
        min_resource=min_resource,
        time_decay=time_decay)

    for i in range(0, workload.shape[1]):
        recommender.update(memory_usage=workload[0, i],
                           cpu_usage=workload[1, i],
                           timestamp=time[i])
        recommendation = recommender.recommender()
        recommendations_memory.append(recommendation[[0, 2, 4]].tolist())
        recommendations_cpu.append(recommendation[[1, 3, 5]].tolist())
    return np.array(recommendations_memory), np.array(recommendations_cpu)


# outpu units
# memory in Megabytes
# cpu in Millicores
# time in seconds
for namespace, pods in cluster.items():
    total_pod_count = len(pods)
    pod_number = 0
    for pod_name, contents in pods.items():

        # ======== fetch information of the pods ========
        workload = contents['workload']
        time = contents['time']
        request_memory = contents['requests']['memory']
        request_cpu = contents['requests']['cpu']
        limit_memory = contents['limits']['memory']
        limit_cpu = contents['limits']['cpu']
        request_density_memory = np.trapz(
            np.ones(time.shape)*request_memory, time
            )
        request_density_cpu = np.trapz(
            np.ones(time.shape)*request_cpu, time
            )

        # ======== usage stats ========
        avg_usage_meomry = int(np.average(workload[0]))
        avg_usage_cpu = int(np.average(workload[1]))
        usage_density_memory = np.trapz(workload[0], time)
        usage_density_cpu = np.trapz(workload[1], time)

        # ======== predictions ========
        # ---- max usage ----
        if margin:
            margin_fraction = 0.15
        else:
            margin_fraction = 0
        request_max_usage_memory = np.max(workload[0])
        request_max_usage_cpu = np.max(workload[1])
        request_max_usage_memory *= (1 + margin_fraction)
        request_max_usage_cpu *= (1 + margin_fraction)
        request_max_usage_memory = int(request_max_usage_memory)
        request_max_usage_cpu = int(request_max_usage_cpu)
        # ---- builtin ----
        request_builtin_per_timestep_memory, request_builtin_per_timestep_cpu =\
            builtin(workload=workload, time=time) # noqa
        request_builtin_final_memory = request_builtin_per_timestep_memory[-1]
        request_builtin_final_cpu = request_builtin_per_timestep_cpu[-1]
        # TODO
        # ---- lstm ----
        # TODO
        # ---- hw ----
        # TODO

        # ======== slacks ========
        # ---- actual usage slacks ----
        # slack over timestep
        slack_usage_per_timestep_memory = request_memory - workload[0]
        slack_usage_per_timestep_cpu = request_cpu - workload[1]
        slack_usage_per_timestep_memory[
            slack_usage_per_timestep_memory < 0] = 0
        slack_usage_per_timestep_cpu[
            slack_usage_per_timestep_cpu < 0] = 0
        # compute area under the curve of slack time
        slack_usage_density_memory = np.trapz(
            slack_usage_per_timestep_memory, time)
        slack_usage_density_cpu = np.trapz(slack_usage_per_timestep_cpu, time)

        # ---- max usage slacks ----
        # slack from the max usage
        slack_max_per_timestep_memory =\
            request_max_usage_memory - workload[0]
        slack_max_per_timestep_cpu =\
            request_max_usage_cpu - workload[1]
        slack_max_per_timestep_memory[
            slack_max_per_timestep_memory < 0] = 0
        slack_max_per_timestep_cpu[slack_max_per_timestep_cpu < 0] = 0
        # compute area under the curve of slack time
        slack_max_density_memory = np.trapz(
            slack_max_per_timestep_memory, time)
        slack_max_density_cpu = np.trapz(slack_max_per_timestep_cpu, time)

        # ---- builtin slacks ----
        # slack from the builtin
        slack_builtin_per_timestep_memory =\
            request_builtin_final_memory[1] - workload[0]
        slack_builtin_per_timestep_cpu =\
            request_builtin_final_cpu[1] - workload[1]
        slack_builtin_per_timestep_memory[
            slack_builtin_per_timestep_memory < 0] = 0
        slack_builtin_per_timestep_cpu[
            slack_builtin_per_timestep_cpu < 0] = 0
        # compute area under the curve of slack time
        slack_builtin_density_memory = np.trapz(
            slack_builtin_per_timestep_memory, time)
        slack_builtin_density_cpu = np.trapz(
            slack_builtin_per_timestep_cpu, time)

        # ======== overrun ========
        # ---- actual usage overrun ----
        # overrun over timestep
        overrun_usage_per_timestep_memory = workload[0] - request_memory
        overrun_usage_per_timestep_cpu = workload[1] - request_cpu
        overrun_usage_per_timestep_memory[
            overrun_usage_per_timestep_memory < 0] = 0
        overrun_usage_per_timestep_cpu[
            overrun_usage_per_timestep_cpu < 0] = 0
        # compute area under the curve of overrun time
        overrun_usage_density_memory = np.trapz(
            overrun_usage_per_timestep_memory, time)
        overrun_usage_density_cpu = np.trapz(
            overrun_usage_per_timestep_cpu, time)

        # ---- max usage overruns ----
        # overrun from the max usage
        overrun_max_per_timestep_memory =\
            workload[0] - request_max_usage_memory
        overrun_max_per_timestep_cpu =\
            workload[1] - request_max_usage_cpu
        overrun_max_per_timestep_memory[
            overrun_max_per_timestep_memory < 0] = 0
        overrun_max_per_timestep_cpu[overrun_max_per_timestep_cpu < 0] = 0
        # compute area under the curve of overrun time
        overrun_max_density_memory = np.trapz(
            overrun_max_per_timestep_memory, time)
        overrun_max_density_cpu = np.trapz(overrun_max_per_timestep_cpu, time)

        # ---- builtin overruns ----
        # overrun from the builtin
        overrun_builtin_per_timestep_memory =\
            workload[0] - request_builtin_final_memory[1]
        overrun_builtin_per_timestep_cpu =\
            workload[1] - request_builtin_final_cpu[1]
        overrun_builtin_per_timestep_memory[
            overrun_builtin_per_timestep_memory < 0] = 0
        overrun_builtin_per_timestep_cpu[
            overrun_builtin_per_timestep_cpu < 0] = 0
        # compute area under the curve of overrun time
        overrun_builtin_density_memory = np.trapz(
            overrun_builtin_per_timestep_memory, time)
        overrun_builtin_density_cpu = np.trapz(
            overrun_builtin_per_timestep_cpu, time)

        # ---------- update the dictionary entries ----------
        cluster[namespace][pod_name].update({

            # -------- usages --------
            # real requests
            'request_memory': request_memory,
            'request_cpu': request_cpu,
            # aveg usage
            'avg_usage_memory': avg_usage_meomry,
            'avg_usage_cpu': avg_usage_cpu,
            'usage_density_memory': usage_density_memory,
            'usage_density_cpu': usage_density_cpu,
            'request_density_memory': request_density_memory,
            'request_density_cpu': request_density_cpu,

            # -------- predictions --------
            # max usage
            'request_max_usage_memory': request_max_usage_memory,
            'request_max_usage_cpu': request_max_usage_cpu,
            # builtin
            'request_builtin_per_timestep_memory':\
            request_builtin_per_timestep_memory,
            'request_builtin_per_timestep_cpu':\
            request_builtin_per_timestep_cpu,
            'request_builtin_final_memory': request_builtin_final_memory,
            'request_builtin_final_cpu': request_builtin_final_cpu,

            # -------- slacks --------
            # slack stats usage
            'slack_usage_per_timestep_memory': slack_usage_per_timestep_memory,
            'slack_usage_per_timestep_cpu': slack_usage_per_timestep_cpu,
            'slack_usage_density_memory': slack_usage_density_memory,
            'slack_usage_density_cpu': slack_usage_density_cpu,
            # slack stats max
            'slack_max_per_timestep_memory': slack_max_per_timestep_memory,
            'slack_max_per_timestep_cpu': slack_max_per_timestep_cpu,
            'slack_max_density_memory': slack_max_density_memory,
            'slack_max_density_cpu': slack_max_density_cpu,
            # slack stats builtin
            'slack_builtin_per_timestep_memory':\
            slack_builtin_per_timestep_memory,
            'slack_builtin_per_timestep_cpu': slack_builtin_per_timestep_cpu,
            'slack_builtin_density_memory': slack_builtin_density_memory,
            'slack_builtin_density_cpu': slack_builtin_density_cpu,

            # -------- overrun --------
            # overrun stats usage
            'overrun_usage_per_timestep_memory':\
            overrun_usage_per_timestep_memory,
            'overrun_usage_per_timestep_cpu': overrun_usage_per_timestep_cpu,
            'overrun_usage_density_memory': overrun_usage_density_memory,
            'overrun_usage_density_cpu': overrun_usage_density_cpu,
            # overrun stats max
            'overrun_max_per_timestep_memory': overrun_max_per_timestep_memory,
            'overrun_max_per_timestep_cpu': overrun_max_per_timestep_cpu,
            'overrun_max_density_memory': overrun_max_density_memory,
            'overrun_max_density_cpu': overrun_max_density_cpu,
            # overrun stats builtin
            'overrun_builtin_per_timestep_memory':\
            overrun_builtin_per_timestep_memory,
            'overrun_builtin_per_timestep_cpu':\
            overrun_builtin_per_timestep_cpu,
            'overrun_builtin_density_memory': overrun_builtin_density_memory,
            'overrun_builtin_density_cpu': overrun_builtin_density_cpu,
        })
        print(f"pod {pod_number} created out of {total_pod_count}")
        pod_number += 1


analysis_path = os.path.join(
    ANALYSIS_CONTAINERS_PATH, f"{cluster_name}.pickle")
with open(analysis_path, 'wb') as out_file:
    pickle.dump(cluster, out_file)
