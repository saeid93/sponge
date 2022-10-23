"""
Iterate through all possible combination
of models and servers
"""

import os
import time
import json
from urllib import response
import yaml
import click
import sys
from PIL import Image
import numpy as np
import csv
import re

from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader
import subprocess
from prom import get_cpu_usage_count, get_memory_usage
from datasets import load_dataset
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from kubernetes import config , client
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

from barazmoon import MLServerBarAzmoon
from barazmoon.twitter import twitter_workload_generator

timeout = 10

def get_pod_name(node_name: str, namespace='default'):
    pod_regex = f"{node_name}.*"
    try:
        config.load_kube_config()
        c = Configuration().get_default_copy()
    except AttributeError:
        c = Configuration()
        c.assert_hostname = False
    Configuration.set_default(c)
    core_v1 = core_v1_api.CoreV1Api()
    ret = core_v1.list_namespaced_pod(namespace)
    pod_names = []
    for i in ret.items:
        pod_name=i.metadata.name
        if re.match(pod_regex, pod_name) and 'svc' not in pod_name:
            pod_names.append(pod_name)
            return pod_names
    return []

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..', '..')))

# import experiments.utils.constants import
from experiments.utils.constants import (
    PIPLINES_PATH,
    NODE_PROFILING_RESULTS_PATH,
    NODE_PROFILING_CONFIGS_PATH,
    NODE_PROFILING_RESULTS_PATH
)

KEY_CONFIG_FILENAME = 'key_config_mapper.csv'

def key_config_mapper(
    pipeline_name: str, node_name: str, cpu_request: str,
    memory_request: str, model_variant: str, max_batch_size: str,
    max_batch_time: str, start_day: int, end_day: int, replica: int):
    file_path = os.path.join(
        NODE_PROFILING_RESULTS_PATH, KEY_CONFIG_FILENAME)
    header = [
        'experiment_id','pipeline_name', 'node_name',
        'model_variant', 'cpu_request',
        'memory_request', 'max_batch_size',
        'max_batch_time', 'start_day', 'end_day', 'replicas']
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
        experiment_id = 0
    else:
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
        experiment_id = line_count
    row = {
        'experiment_id': experiment_id,
        'pipeline_name': pipeline_name,
        'model_variant': model_variant,
        'node_name': node_name,
        'cpu_request': cpu_request,
        'memory_request': memory_request,
        'max_batch_size': max_batch_size,
        'max_batch_time': max_batch_time,
        'start_day': start_day,
        'end_day': end_day,
        'replicas': replica
        }
    with open(file_path, 'a') as row_writer:
        dictwriter_object = csv.DictWriter(row_writer, fieldnames=header)
        dictwriter_object.writerow(row)
        row_writer.close()
    return experiment_id

def experiments(pipeline_name: str, node_name: str,
                config: dict, node_path: str, data_type: str):
    model_vairants = config['model_vairants']
    max_batch_sizes = config['max_batch_size']
    max_batch_times = config['max_batch_time']
    cpu_requests = config['cpu_request']
    memory_requests = config["memory_request"]
    replica = config['replicas']
    workload_config = config['workload_config']
    start_day = workload_config['start_day']
    end_day = workload_config['end_day']
    repetition = config['repetition']
    # Better solution instead of nested for loops
    # TODO also add the random - maybe just use Tune
    for model_variant in model_vairants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replica:
                            setup_node(
                                node_name=node_name,
                                cpu_request=cpu_request,
                                memory_request=memory_request,
                                model_variant=model_variant,
                                max_batch_size=max_batch_size,
                                max_batch_time=max_batch_time,
                                replica=replica,
                                node_path=node_path
                            )
                            time.sleep(timeout) # TODO better validation -> some request
                            for _ in range(repetition):
                                experiment_id = key_config_mapper(
                                    pipeline_name=pipeline_name,
                                    node_name=node_name,
                                    cpu_request=cpu_request,
                                    memory_request=memory_request,
                                    model_variant=model_variant,
                                    max_batch_size=max_batch_size,
                                    max_batch_time=max_batch_time,
                                    replica=replica)
                                start_time = time.time()
                                responses = list()
                                responses = load_test(
                                    node_name=node_name,
                                    data_type=data_type,
                                    node_path=node_path,
                                    start_day=start_day,
                                    end_day=end_day)
                                time.sleep(timeout) # TODO better validation -> some request
                                end_time = time.time()
                                time.sleep(timeout) # TODO better validation -> some request
                                save_report(
                                    experiment_id=experiment_id,
                                    responses = responses,
                                    node_name=node_name,
                                    start_time=start_time,
                                    end_time=end_time) # TODO id system for the experiments
                            remove_node(node_name=node_name)

def setup_node(node_name: str, cpu_request: str,
               memory_request: str, model_variant: str, max_batch_size: str,
               max_batch_time: str, replica: int, node_path: str):
    svc_vars = {
        "name": node_name,
        "cpu_request": cpu_request,
        "memory_request": memory_request,
        "cpu_limit": cpu_request,
        "memory_limit": memory_request,
        "model_vairant": model_variant,
        "max_batch_size": max_batch_size,
        "max_batch_time": max_batch_time,
        "replicas": replica
        }
    environment = Environment(
        loader=FileSystemLoader(node_path))
    svc_template = environment.get_template('node-template.yaml')
    content = svc_template.render(svc_vars)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)

def load_test(node_name: str, data_type: str,
              node_path: str, start_day: int,
              end_day: int):
    # load sample data
    if data_type == 'audio':
        input_sample_path = os.path.join(
            node_path, 'input-sample.json'
        )
        input_sample_shape_path = os.path.join(
            node_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = json.load(openfile)
        with open(input_sample_shape_path, 'r') as openfile:
            data_shape = json.load(openfile)
            data_shape = data_shape['data_shape']
    elif data_type == 'text':
        input_sample_path = os.path.join(
            node_path, 'input-sample.txt'
        )
        input_sample_shape_path = os.path.join(
            node_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = [openfile.read()]
        with open(input_sample_shape_path, 'r') as openfile:
            data_shape = json.load(openfile)
            data_shape = data_shape['data_shape']
    elif data_type == 'image':
        input_sample_path = os.path.join(
            node_path, 'input-sample.JPEG'
        )
        input_sample_shape_path = os.path.join(
            node_path, 'input-sample-shape.json'
        )
        data = np.array(Image.open(input_sample_path)).tolist()
        with open(input_sample_shape_path, 'r') as openfile:
            data_shape = json.load(openfile)
            data_shape = data_shape['data_shape']
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    # load test on the server
    gateway_endpoint = "localhost:32000"
    namespace = "default"
    endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{node_name}/v2/models/infer"
    workload = twitter_workload_generator(f"{start_day}-{end_day}")
    load_tester = MLServerBarAzmoon(
        endpoint=endpoint,
        http_method='post',
        workload=workload,
        data=data,
        data_shape=data_shape,
        data_type=data_type)
    load_tester.start()
    responses = load_tester.get_responses()
    return responses

def remove_node(node_name):
    os.system(f"kubectl delete seldondeployment {node_name} -n default")


def save_report(experiment_id: int,
                responses: str,
                node_name: str,
                start_time: float,
                end_time: float):
    results = {
        'cpu_usage': [],
        'time_cpu': [],
        'memory_usage': [],
        'time_memory': [],
        'responses': responses,
        'latency': [],
        'throughput': []
    }
    save_path = os.path.join(
        NODE_PROFILING_RESULTS_PATH, f"{experiment_id}.json") # TODO experiments id system
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # TODO postprocess of results
    # TODO consider repetition_id
    # TODO save results

    end_time = time.time()
    duration = (end_time - start_time)//60 + 1
    print(duration)
    # TODO add list of pods in case of replicas
    pod_name = get_pod_name(node_name)[0]
    cpu_usage, time_cpu = get_cpu_usage_count(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name)
    memory_usage, time_memory = get_memory_usage(
        pod_name=pod_name, namespace="default",
        container=node_name, duration=int(duration), need_max=False)
    results['cpu_usage'] = cpu_usage
    results['memory_usage'] = memory_usage
    results['time_cpu'] = time_cpu
    results['time_memory'] = time_memory
    with open(save_path, "w") as outfile:
        outfile.write(json.dumps(results))
    print(f'results have been sucessfully saved in:\n{save_path}')

@click.command()
@click.option(
    '--config-name', required=True, type=str, default='config_static')
def main(config_name: str):
    config_path = os.path.join(
        NODE_PROFILING_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    pipeline_name = config['pipeline_name']
    node_name = config['node_name']
    data_type = config['data_type']
    node_path = os.path.join(
        PIPLINES_PATH,
        pipeline_name,
        'seldon-core-version',
        'nodes',
        node_name
    )
    experiments(
        pipeline_name=pipeline_name,
        node_name=node_name,
        config=config,
        node_path=node_path,
        data_type=data_type
        )

if __name__ == "__main__":
    main()

# ===========================================

# def change_names(names):
#     return_names = []
#     for name in names:
#         return_names.append(name.replace("_", "-"))
#     return return_names


# def extract_node_timer(json_data : dict):
#     keys = list(json_data.keys())
#     nodes = []
#     sir_names = ["arrival_"]
#     for name in sir_names:
#         for key in keys:
#             if name in key:
#                 nodes.append(key.replace(name, ""))

#     return_nodes = change_names(nodes)
#     return_timer = {}
#     for node in nodes:
#         return_timer[node] = json_data["serving_" + node] - json_data["arrival_" + node]
#     e2e_lats = json_data[keys[-1]] - json_data[keys[0]]

#     return return_nodes, return_timer, e2e_lats
    
# def load_test(
#     pipeline_name: str,
#     inputs: Dict[str, Any],
#     node_1_model, 
#     node_2_model,
#     n_items: int,
#     n_iters = 40
#     ):
#     start = time.time()
#     gateway_endpoint="localhost:32000"
#     deployment_name = pipeline_name 
#     namespace = "default"
#     num_nodes = pipeline_name.split("-").__len__()
#     e2e_lats = []
#     node_latencies = [[] for _ in range(num_nodes)]
#     cpu_usages = [[] for _ in range(num_nodes) ]
#     memory_usages = [[] for _ in range(num_nodes) ]
#     sc = SeldonClient(
#         gateway_endpoint=gateway_endpoint,
#         gateway="istio",
#         transport="rest",
#         deployment_name=deployment_name,
#         namespace=namespace)

#     time.sleep(CHECK_TIMEOUT)
#     for iter in range(n_iters):
#         response = sc.predict(
#             data=inputs
#         )

#         if response.success:
#             json_data_timer = response.response['jsonData']['time']
#             return_nodes, return_timer, e2e_lat = extract_node_timer(json_data_timer)
#             for i , name in enumerate(return_nodes):
#                 cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
#                 memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, 1))
#                 e2e_lats.append(e2e_lat)
#             for i, time_ in enumerate(return_timer.keys()):
#                 node_latencies[i].append(return_timer[time_])

#         else:
#             pp.pprint(response.msg)
#         print(iter)
#     time.sleep(CHECK_TIMEOUT)
#     total_time = int((time.time() - start)//60)
#     for i , name in enumerate(return_nodes):
#         cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
#         memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, total_time, True))
#     models = node_1_model + "*" + node_2_model + "*"
#     with open(save_path+"/cpu.txt", "a") as cpu_file:
#         cpu_file.write(f"usage of {models} {pipeline_name} is {cpu_usages} \n")

#     with open(save_path+"/memory.txt", 'a') as memory_file:
#         memory_file.write(f"usage of {models} {pipeline_name} is {memory_usages} \n")


#     with open(save_path+"/node-latency.txt", "a") as infer:
#         infer.write(f"lats of {models} {pipeline_name} is {node_latencies} \n")
    
#     with open(save_path+"/ee.txt", "a") as s:
#         s.write(f"eelat of {models} {pipeline_name} is {e2e_lats} \n")
    

# for node_1_model in node_1_models:
#     for node_2_model in node_2_models:
#         pipeline_name = prune_name(node_1_model, 8) + "-" +\
#             prune_name(node_2_model, 8)
#         start_time = time.time()
#         while True:
#             setup_pipeline(
#                 node_1_model=node_1_model,
#                 node_2_model=node_2_model,
#                 template=TEMPLATE, pipeline_name=pipeline_name)
#             time.sleep(CHECK_TIMEOUT)
#             command = ("kubectl rollout status deploy/$(kubectl get deploy"
#                     f" -l seldon-deployment-id={pipeline_name} -o"
#                     " jsonpath='{.items[0].metadata.name}')")
#             time.sleep(CHECK_TIMEOUT)
#             p = subprocess.Popen(command, shell=True)
#             try:
#                 p.wait(RETRY_TIMEOUT)
#                 break
#             except subprocess.TimeoutExpired:
#                 p.kill()
#                 print("corrupted pipeline, should be deleted ...")
#                 remove_pipeline(pipeline_name=pipeline_name)
#                 print('waiting to delete ...')
#                 time.sleep(DELETE_WAIT)

