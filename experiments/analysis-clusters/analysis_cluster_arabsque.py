import pickle
import os
import sys
from collections import Counter
import json
from copy import deepcopy

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    ANALYSIS_CONTAINERS_PATH,
    FINAL_STATS_PATH,
)


def cluster_operations(clusters_names):

    clusters = {}
    for cluster_name in clusters_names:
        # analysis containers path
        analysis_containers_path = os.path.join(
            ANALYSIS_CONTAINERS_PATH, f"{cluster_name}.pickle")
        with open(analysis_containers_path, 'rb') as in_file:
            cluster = pickle.load(in_file)
        clusters[cluster_name] = cluster

    # # analysis clusters path
    # analysis_clusters_path = os.path.join(
    #     ANALYSIS_CLUSTERS_PATH)

    keys = [
        # -------- usages --------
        # real requests
        'request_memory',
        'request_cpu',
        # aveg usage
        'avg_usage_memory',
        'avg_usage_cpu',
        'usage_density_memory',
        'usage_density_cpu',
        'request_density_memory',
        'request_density_cpu',
        # -------- slacks --------
        # slack stats usage
        'slack_usage_density_memory',
        'slack_usage_density_cpu',
        # slack stats max
        'slack_max_density_memory',
        'slack_max_density_cpu',
        # slack stats builtin
        'slack_builtin_density_memory',
        'slack_builtin_density_cpu',
        # -------- overrun --------
        # overrun builtin
        'overrun_usage_density_memory',
        'overrun_usage_density_cpu',
        'overrun_max_density_memory',
        'overrun_max_density_cpu',
        'overrun_builtin_density_memory',
        'overrun_builtin_density_cpu',
    ]

    clusters_stats = {}
    # cumulative stats
    for cluster_name, cluster in clusters.items():
        sum_dict = {k: 0 for k in keys}
        total_num_pods = 0
        for namespace, pods in cluster.items():
            num_pods = len(pods)
            sum_dict = Counter(sum_dict)
            for _, contents in pods.items():
                contents_filtered = dict(zip(
                    keys, [contents[k] for k in keys]))
                contents_filtered = Counter(contents_filtered)
                sum_dict += contents_filtered
            total_num_pods += num_pods
        sum_dict = dict(sum_dict)
        # update missing keys
        missing_keys = list(set(keys) - set(sum_dict.keys()))
        missing_enteries = {k: 0 for k in missing_keys}
        sum_dict.update(missing_enteries)
        sum_dict['total_num_pods'] = total_num_pods
        clusters_stats[cluster_name] = deepcopy(sum_dict)

    # Fix units
    # input:
    # cpu - Millicores
    # memory - Megabytes
    # time - seconds
    # output:
    # cpu - cores
    # memory - Gigabytes
    # time - hours
    memory_ratio = 1000
    cpu_ratio = 1000
    time_ratio = 3600
    for cluster_name, cluster in clusters_stats.items():
        cluster['request_memory'] /= memory_ratio
        cluster['request_cpu'] /= cpu_ratio
        cluster['avg_usage_memory'] /= memory_ratio
        cluster['avg_usage_cpu'] /= cpu_ratio
        cluster['usage_density_memory'] /= (memory_ratio * time_ratio)
        cluster['usage_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['request_density_memory'] /= (memory_ratio * time_ratio)
        cluster['request_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['slack_usage_density_memory'] /= (memory_ratio * time_ratio)
        cluster['slack_usage_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['slack_max_density_memory'] /= (memory_ratio * time_ratio)
        cluster['slack_max_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['slack_builtin_density_memory'] /= (memory_ratio * time_ratio)
        cluster['slack_builtin_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['overrun_usage_density_memory'] /= (memory_ratio * time_ratio)
        cluster['overrun_usage_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['overrun_max_density_memory'] /= (memory_ratio * time_ratio)
        cluster['overrun_max_density_cpu'] /= (cpu_ratio * time_ratio)
        cluster['overrun_builtin_density_memory'] /= (
            memory_ratio * time_ratio)
        cluster['overrun_builtin_density_cpu'] /= (cpu_ratio * time_ratio)

    # Resource Usage - Engine - stat 1
    # dataframe
    stat_11 = {}
    for cluster_name, cluster in clusters_stats.items():
        stat_11_entery = {}
        stat_11_entery['avg_request_memory'] =\
            round(cluster['request_memory'] / cluster['total_num_pods'], 2)
        stat_11_entery['avg_request_cpu'] =\
            round(cluster['request_cpu'] / cluster['total_num_pods'], 2)
        stat_11_entery['avg_usage_memory'] =\
            round(cluster['avg_usage_memory'] / cluster['total_num_pods'], 2)
        stat_11_entery['avg_usage_cpu'] =\
            round(cluster['avg_usage_cpu'] / cluster['total_num_pods'], 2)
        stat_11[cluster_name] = deepcopy(stat_11_entery)

    stat_12 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_12_entery = {}
        stat_12_entery['memory'] = int(cluster['usage_density_memory'])
        stat_12_entery['fraction_memory'] =\
            round(cluster['usage_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_12_entery['cpu'] = int(cluster['usage_density_cpu'])
        stat_12_entery['fraction_cpu'] =\
            round(cluster['usage_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_12[cluster_name] = deepcopy(stat_12_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    usage_density_memory_total = 0
    usage_density_cpu_total = 0
    for cluster_name, cluster in stat_12.items():
        usage_density_memory_total += cluster['memory']
        usage_density_cpu_total += cluster['cpu']
    stat_12.update(
        {'total': {
            'memory': usage_density_memory_total,
            'cpu': usage_density_cpu_total,
            'fraction_memory': round(
                usage_density_memory_total/request_density_memory_total, 2),
            'fraction_cpu': round(
                usage_density_cpu_total/request_density_cpu_total, 2)
        }})
    with open(os.path.join(FINAL_STATS_PATH, 'stat11.json'), 'x') as out_file:
        json.dump(stat_11, out_file, indent=4)
    with open(os.path.join(FINAL_STATS_PATH, 'stat12.json'), 'x') as out_file:
        json.dump(stat_12, out_file, indent=4)

    # Slack Over time - stat 4
    stat_4 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_4_entery = {}
        stat_4_entery['memory'] = int(cluster['slack_usage_density_memory'])
        stat_4_entery['fraction_memory'] =\
            round(cluster['slack_usage_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_4_entery['cpu'] = int(cluster['slack_usage_density_cpu'])
        stat_4_entery['fraction_cpu'] =\
            round(cluster['slack_usage_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_4[cluster_name] = deepcopy(stat_4_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    slack_usage_density_memory_total = 0
    slack_usage_density_cpu_total = 0
    for cluster_name, cluster in stat_4.items():
        slack_usage_density_memory_total += cluster['memory']
        slack_usage_density_cpu_total += cluster['cpu']
    stat_4.update(
        {'total': {
            'memory': slack_usage_density_memory_total,
            'cpu': slack_usage_density_cpu_total,
            'fraction_memory': round(
                slack_usage_density_memory_total/request_density_memory_total,
                2),
            'fraction_cpu': round(
                slack_usage_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat4.json'), 'x') as out_file:
        json.dump(stat_4, out_file, indent=4)

    # Overrun from the request - Stats - stat 5
    stat_5 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_5_entery = {}
        stat_5_entery['memory'] = int(cluster['overrun_usage_density_memory'])
        stat_5_entery['fraction_memory'] =\
            round(cluster['overrun_usage_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_5_entery['cpu'] = int(cluster['overrun_usage_density_cpu'])
        stat_5_entery['fraction_cpu'] =\
            round(cluster['overrun_usage_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_5[cluster_name] = deepcopy(stat_5_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    overrun_usage_density_memory_total = 0
    overrun_usage_density_cpu_total = 0
    for cluster_name, cluster in stat_5.items():
        overrun_usage_density_memory_total += cluster['memory']
        overrun_usage_density_cpu_total += cluster['cpu']
    stat_5.update(
        {'total': {
            'memory': overrun_usage_density_memory_total,
            'cpu': overrun_usage_density_cpu_total,
            'fraction_memory': round(
                overrun_usage_density_memory_total/request_density_memory_total, # noqa
                2),
            'fraction_cpu': round(
                overrun_usage_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat5.json'), 'x') as out_file:
        json.dump(stat_5, out_file, indent=4)

    # Slack with max prediction - stat 6
    stat_6 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_6_entery = {}
        stat_6_entery['memory'] = int(cluster['slack_max_density_memory'])
        stat_6_entery['fraction_memory'] =\
            round(cluster['slack_max_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_6_entery['cpu'] = int(cluster['slack_max_density_cpu'])
        stat_6_entery['fraction_cpu'] =\
            round(cluster['slack_max_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_6[cluster_name] = deepcopy(stat_6_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    slack_max_density_memory_total = 0
    slack_max_density_cpu_total = 0
    for cluster_name, cluster in stat_6.items():
        slack_max_density_memory_total += cluster['memory']
        slack_max_density_cpu_total += cluster['cpu']
    stat_6.update(
        {'total': {
            'memory': slack_max_density_memory_total,
            'cpu': slack_max_density_cpu_total,
            'fraction_memory': round(
                slack_max_density_memory_total/request_density_memory_total,
                2),
            'fraction_cpu': round(
                slack_max_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat6.json'), 'x') as out_file:
        json.dump(stat_6, out_file, indent=4)

    # Overrun with max prediction - stat 7
    stat_7 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_7_entery = {}
        stat_7_entery['memory'] = int(cluster['overrun_max_density_memory'])
        stat_7_entery['fraction_memory'] =\
            round(cluster['overrun_max_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_7_entery['cpu'] = int(cluster['overrun_max_density_cpu'])
        stat_7_entery['fraction_cpu'] =\
            round(cluster['overrun_max_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_7[cluster_name] = deepcopy(stat_7_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    overrun_max_density_memory_total = 0
    overrun_max_density_cpu_total = 0
    for cluster_name, cluster in stat_7.items():
        overrun_max_density_memory_total += cluster['memory']
        overrun_max_density_cpu_total += cluster['cpu']
    stat_7.update(
        {'total': {
            'memory': overrun_max_density_memory_total,
            'cpu': overrun_max_density_cpu_total,
            'fraction_memory': round(
                overrun_max_density_memory_total/request_density_memory_total,
                2),
            'fraction_cpu': round(
                overrun_max_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat7.json'), 'x') as out_file:
        json.dump(stat_7, out_file, indent=4)

    # Slack with Builtin prediction - stat 8
    stat_8 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_8_entery = {}
        stat_8_entery['memory'] = int(cluster['slack_builtin_density_memory'])
        stat_8_entery['fraction_memory'] =\
            round(cluster['slack_builtin_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_8_entery['cpu'] = int(cluster['slack_builtin_density_cpu'])
        stat_8_entery['fraction_cpu'] =\
            round(cluster['slack_builtin_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_8[cluster_name] = deepcopy(stat_8_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    slack_builtin_density_memory_total = 0
    slack_builtin_density_cpu_total = 0
    for cluster_name, cluster in stat_8.items():
        slack_builtin_density_memory_total += cluster['memory']
        slack_builtin_density_cpu_total += cluster['cpu']
    stat_8.update(
        {'total': {
            'memory': slack_builtin_density_memory_total,
            'cpu': slack_builtin_density_cpu_total,
            'fraction_memory': round(
                slack_builtin_density_memory_total/request_density_memory_total,
                2),
            'fraction_cpu': round(
                slack_builtin_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat8.json'), 'x') as out_file:
        json.dump(stat_8, out_file, indent=4)

    # Overrun with Builtin prediction - stat 9
    stat_9 = {}
    request_density_memory_total = 0
    request_density_cpu_total = 0
    for cluster_name, cluster in clusters_stats.items():
        stat_9_entery = {}
        stat_9_entery['memory'] = int(cluster['overrun_builtin_density_memory'])
        stat_9_entery['fraction_memory'] =\
            round(cluster['overrun_builtin_density_memory'] / cluster[
                        'request_density_memory'], 2)
        stat_9_entery['cpu'] = int(cluster['overrun_builtin_density_cpu'])
        stat_9_entery['fraction_cpu'] =\
            round(cluster['overrun_builtin_density_cpu'] / cluster[
                'request_density_cpu'], 2)
        stat_9[cluster_name] = deepcopy(stat_9_entery)
        request_density_memory_total += cluster['request_density_memory']
        request_density_cpu_total += cluster['request_density_cpu']

    overrun_builtin_density_memory_total = 0
    overrun_builtin_density_cpu_total = 0
    for cluster_name, cluster in stat_9.items():
        overrun_builtin_density_memory_total += cluster['memory']
        overrun_builtin_density_cpu_total += cluster['cpu']
    stat_9.update(
        {'total': {
            'memory': overrun_builtin_density_memory_total,
            'cpu': overrun_builtin_density_cpu_total,
            'fraction_memory': round(
                overrun_builtin_density_memory_total/request_density_memory_total,
                2),
            'fraction_cpu': round(
                overrun_builtin_density_cpu_total/request_density_cpu_total, 2)
        }})

    with open(os.path.join(FINAL_STATS_PATH, 'stat9.json'), 'x') as out_file:
        json.dump(stat_9, out_file, indent=4)

    # for namespace, stats in clusters_stats.items():
    #     for metric, metric_value in stats.items():
    #         if 'memory' in metric:
    #             # unit Gb * hour
    #             clusters_stats[namespace][metric] = metric_value * (1/3600)
    #         if 'cpu' in metric:
    #             # unit core * hour
    #             clusters_stats[namespace][
    #                 metric] = metric_value * (1/1e3 * 1/3600)
    # a = 1


def plot_histograms(clusters_names):
    pass


# options: engine-july-all | engine-top-ten | portfolio-july-all
#          portfolio-top-ten


# cluster_names = ['engine-july-all', 'engine-top-ten',
#                  'portfolio-july-all', 'portfolio-top-ten']

# per cluster
clusters_names = ["engine-july-all", "portfolio-july-all"]
# clusters_names = ["engine-top-ten"]
cluster_operations(clusters_names)






# # all at once
# for cluster_name in cluster_names:
#     cluster_operations(cluster_name)
