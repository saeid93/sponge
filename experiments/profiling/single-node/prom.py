import requests
from typing import List, Dict

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta


PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(
    url ="http://localhost:30090", disable_ssl=True)


def prom_response_postprocess(response):
    try:
        response = response.json()['data']['result']
    except:
        print(response.json())
        exit()
    plot_values = [[] for _ in range(len(response))]
    times = [[] for _ in range(len(response))]

    for val in range(len(response)):
        data = response[val]['values']
        for d in data:
            plot_values[val].append((float(d[1])))
            times[val].append(float(d[0]))

    return plot_values[0] , times[0]


def get_memory_usage(pod_name: str, namespace: str,
                     container: str, duration: int,
                     need_max: bool = False):
    query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m:1s]"
    if need_max:
        query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m])"
    
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)

def get_cpu_usage_count(pod_name: str, namespace: str,
                        container: str, duration: int):
    query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)

def get_cpu_usage_rate(pod_name: str, namespace: str,
                       container: str, duration: int, rate=120):
    query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)

def get_cpu_throttled_count(pod_name: str, namespace: str,
                        container: str, duration: int):
    query = f"container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)

def get_cpu_throttled_rate(pod_name: str, namespace: str,
                       container: str, duration: int, rate=120):
    query = f"rate(container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)


def get_request_per_second(pod_name, namespace, container, duration, rate=120):
    query = f"rate(model_infer_request_duration_count{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
    response = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    return prom_response_postprocess(response)
