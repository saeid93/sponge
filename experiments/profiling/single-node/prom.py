import pandas as pd
import numpy as np
import requests
from matplotlib import pyplot as plt

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta


PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(
    url ="http://localhost:30090", disable_ssl=True)


def get_memory_usage(pod_name: str, namespace: str,
                     container: str, duration: int,
                     need_max: bool = False):
    # PROMETHEUS = "http://localhost:30090"
    # prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m:1s]"
    if need_max:
        query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m])"
    
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = response_memory_usage.json()['data']['result']
    plot_values = [[] for _ in range(len(values))]
    times = [[] for _ in range(len(values))]
    for val in range(len(values)):
        data = values[val]['values']
        for d in data:
            plot_values[val].append((float(d[1])))
            times[val].append(float(d[0]))
    return plot_values[0], times[0]

def get_cpu_usage(pod_name, namespace, container, duration):
    # PROMETHEUS = "http://localhost:30090"
    # prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
    response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    try:
        values = response_cpu_usage.json()['data']['result']
    except:
        print(response_cpu_usage.json())
        exit()
    plot_values = [[] for _ in range(len(values))]
    times = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['values']
        for d in data:
            plot_values[val].append((float(d[1])))
            times[val].append(float(d[0]))

    return plot_values[0] , times[0]


def get_cpu_usage_rate(pod_name, namespace,container, duration, rate=120):
    # PROMETHEUS = "http://localhost:30090"
    # prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
    response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    try:
        values = response_cpu_usage.json()['data']['result']
    except:
        print(response_cpu_usage.json())
        exit()
    plot_values = [[] for _ in range(len(values))]
    times = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['values']
        for d in data:
            plot_values[val].append((float(d[1])))
            times[val].append(float(d[0]))

    return plot_values[0] , times[0]