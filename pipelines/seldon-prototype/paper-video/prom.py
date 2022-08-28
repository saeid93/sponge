from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import requests
PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)


def get_memory_usage(pod_name, name_space,end, need_max = False):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='video-yolo', name_space='{name_space}'}}"
    if need_max:
        query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='video-yolo'}}[{end}m])"

    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = response_memory_usage.json()['data']['result']
    print(response_memory_usage.json())
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['value']
        plot_values[val].append((float(data[1])))
    return plot_values[0][0]

def get_cpu_usage(pod_name, name_space,):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{name_space}', container='video-yolo'}}"
    response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    try:
        values = response_cpu_usage.json()['data']['result']
    except:
        print(response_cpu_usage.json())
        exit()
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['value']
        plot_values[val].append((float(data[1])))
    return plot_values[0][0]

print(get_memory_usage("yolov5nresnet18", "default", 3))