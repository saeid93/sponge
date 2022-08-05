

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import requests
PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

def get_inference_duration():
    query = f"nv_inference_compute_infer_duration_us"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})


def get_comput_input_duration():
    query = f"nv_inference_compute_input_duration_us"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})


def get_computer_output_duration():
    query = f"nv_inference_compute_output_duration_us"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})



def get_inference_count():
    query = f"nv_inference_count"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
 



def get_inference_failure():
    query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}', image!='', container_name!='POD'}}[30s])[2m:30s]"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})


def get_memory_usage(pod_name):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"rate(container_memory_working_set_bytes{{pod='{pod_name}'}}[1m])[3m:1m]"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = response_memory_usage.json()['data']['result']
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['values']
        for dat in data:
            plot_values[val].append((float(dat[1])))
    return plot_values

def get_cpu_usage(pod_name):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}', image!='', container_name!='POD'}}[1m])[3m:1m]"
    response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})

    values = response_cpu_usage.json()['data']['result']
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['values']
        for dat in data:
            plot_values[val].append((float(dat[1])))
    return plot_values
