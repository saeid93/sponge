import pickle
import os
import sys
from collections import Counter
import json
import matplotlib.pyplot as plt
from copy import deepcopy
from smart_kube.util import Histogram, plot_histogram

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    ANALYSIS_CONTAINERS_PATH,
    FINAL_STATS_PATH,
)


def plot_usage_histogram(cluster_name):
    cluster = {}

    analysis_containers_path = os.path.join(
        ANALYSIS_CONTAINERS_PATH, f"{cluster_name}.pickle")
    with open(analysis_containers_path, 'rb') as in_file:
        cluster = pickle.load(in_file)

    max_value = 95
    first_bucket_size = 5
    cpu_usage_histogram = Histogram(
        max_value=max_value,
        first_bucket_size=first_bucket_size,
        time_decay=False,
        ratio=1
    )

    memory_usage_histogram = Histogram(
        max_value=max_value,
        first_bucket_size=first_bucket_size,
        time_decay=False,
        ratio=1
    )

    cpu_usage_slack_histogram = Histogram(
        max_value=max_value,
        first_bucket_size=first_bucket_size,
        time_decay=False,
        ratio=1
    )


    for _, pods in cluster.items():
        for _, contents in pods.items():
            memory_usage_sample = contents['usage_density_memory']/contents['request_density_memory'] * 100 # noqa
            memory_usage_histogram.add_sample(value=memory_usage_sample, weight=1)

            cpu_usage_sample = contents['usage_density_cpu']/contents['request_density_memory'] * 100 # noqa
            cpu_usage_histogram.add_sample(value=cpu_usage_sample, weight=1)

    fig1 = plot_histogram(
        histogram=memory_usage_histogram,
        title='memory usage histogram',
        x_label='usage percentage')
    fig1.savefig(f'{cluster_name}-memory_usage_histogram.png')
    plt.close()

    fig2 = plot_histogram(
        histogram=cpu_usage_histogram,
        title='cpu usage histogram',
        x_label='usage percentage')
    fig2.savefig(f'{cluster_name}-cpu_usage_histogram.png')
    plt.close()


# portfolio-july-all | engine-july-all
plot_usage_histogram('engine-july-all')
