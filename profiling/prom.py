

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import requests
PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

def get_inference_duration(model, version):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"nv_inference_compute_infer_duration_us{{model='{model}', version='{version}'}}"
    infer_time_res = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = infer_time_res.json()['data']['result'][-1]['value']
    return_values = float(values[1])   
    return return_values

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


def get_memory_usage(pod_name, name_space,end, rate):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"max_over_time(rate(container_memory_usage_bytes{{pod='{pod_name}', namespace='{name_space}', container='tritonserver'}}[{rate}m])[{end}m:{rate}m])"
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = response_memory_usage.json()['data']['result']
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['value']
        plot_values[val].append((float(data[1])))
    return plot_values[0][0]

def get_cpu_usage(pod_name, name_space, end, rate):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"max_over_time(rate(container_cpu_usage_seconds_total{{pod='{pod_name}', namespace='{name_space}', container='tritonserver'}}[{rate}m])[{end}m:{rate}m])"
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


