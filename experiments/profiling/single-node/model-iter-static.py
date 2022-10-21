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

from typing import List
from jinja2 import Environment, FileSystemLoader
from prom import (
    get_cpu_usage,
    get_cpu_usage_rate,
    get_memory_usage)
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

timeout = 30

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
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..', '..', '..')))

# import experiments.utils.constants import
from experiments.utils.constants import (
    PIPLINES_PATH,
    NODE_PROFILING_CONFIGS_PATH,
    NODE_PROFILING_RESULTS_STATIC_PATH
)

KEY_CONFIG_FILENAME = 'key_config_mapper.csv'

def key_config_mapper(
    pipeline_name: str, node_name: str, cpu_request: str,
    memory_request: str, model_variant: str, max_batch_size: str,
    max_batch_time: str, load: int,
    load_duration: int, series: int, series_meta: str, replica: int):
    file_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH, KEY_CONFIG_FILENAME)
    header = [
        'experiment_id','pipeline_name', 'node_name',
        'model_variant', 'cpu_request',
        'memory_request', 'max_batch_size',
        'max_batch_time', 'load', 'load_duration',
        'series', 'series_meta', 'replicas']
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
        experiment_id = 1
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
        'load': load,
        'load_duration': load_duration,
        'series': series,
        'series_meta': series_meta,
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
    repetition = config['repetition']
    series = config['series']
    series_meta = config['series_meta']
    loads_to_test = workload_config['loads_to_test']
    load_duration = workload_config['load_duration']
    # TODO Better solution instead of nested for loops
    # TODO also add the random - maybe just use Tune
    for model_variant in model_vairants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replica:
                            for load in loads_to_test:

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

                                for rep in range(repetition):
                                    print('-'*25 + f' starting repetition {rep} ' + '-'*25)
                                    print('\n')
                                    if rep != 0: time.sleep(60) # TODO timeout var
                                    experiment_id = key_config_mapper(
                                        pipeline_name=pipeline_name,
                                        node_name=node_name,
                                        cpu_request=cpu_request,
                                        memory_request=memory_request,
                                        model_variant=model_variant,
                                        max_batch_size=max_batch_size,
                                        max_batch_time=max_batch_time,
                                        load=load,
                                        load_duration=load_duration,
                                        series=series,
                                        series_meta=series_meta,
                                        replica=replica)

                                    start_time, end_time, responses = load_test(
                                        node_name=node_name,
                                        data_type=data_type,
                                        node_path=node_path,
                                        load=load,
                                        namespace='default',
                                        load_duration=load_duration)
                                    
                                    print("--------HERE----------")

                                    save_report(
                                        experiment_id=experiment_id,
                                        responses = responses,
                                        node_name=node_name,
                                        start_time=start_time,
                                        end_time=end_time) # TODO id system for the experiments

                                    print("--------THERE----------")

                                time.sleep(timeout) # TODO better validation -> some request
                                remove_node(node_name=node_name)

def setup_node(node_name: str, cpu_request: str,
               memory_request: str, model_variant: str, max_batch_size: str,
               max_batch_time: str, replica: int, node_path: str):
    print('-'*25 + ' setting up the node with following config' + '-'*25)
    print('\n')
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
    pp.pprint(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    time.sleep(timeout) # TODO better validation -> some request
    print('-'*25 + f' waiting {timeout} to make sure the node is up ' + '-'*25)
    print('\n')
    print('-'*25 + f' model pod {timeout} successfuly set up ' + '-'*25)
    print('\n')

def load_test(node_name: str, data_type: str,
              node_path: str,
              load: int, load_duration: int,
              namespace: str='default',):
    start_time = time.time()
    print('-'*25 + f' starting load test ' + '-'*25)
    print('\n')
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
    endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{node_name}/v2/models/infer"
    workload = [load] * load_duration
    load_tester = MLServerBarAzmoon(
        endpoint=endpoint,
        http_method='post',
        workload=workload,
        data=data,
        data_shape=data_shape,
        data_type=data_type)
    load_tester.start()

    print("--------KOSE DONYA----------")

    responses = load_tester.get_responses()

    print("--------SOMEWHERE----------")

    end_time = time.time()
    return start_time, end_time, responses

def remove_node(node_name):
    os.system(f"kubectl delete seldondeployment {node_name} -n default")
    print('-'*50 + f' model pod {timeout} successfuly set up ' + '-'*50)
    print('\n')


def save_report(experiment_id: int,
                responses: str,
                node_name: str,
                start_time: float,
                end_time: float,
                namespace: str = 'default'):
    results = {
        'cpu_usage': [],
        'time_cpu': [],
        'memory_usage': [],
        'time_memory': [],
        'responses': responses,
        'start_time': start_time,
        'end_time': end_time,
        'latency': [],
        'throughput': []
    }
    save_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH, f"{experiment_id}.json") # TODO experiments id system
    duration = (end_time - start_time)//60 + 1
    # print(duration)
    # TODO add list of pods in case of replicas
    pod_name = get_pod_name(node_name=node_name, namespace=namespace)[0]
    cpu_usage, time_cpu = get_cpu_usage(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name)
    cpu_usage, time_cpu = get_cpu_usage_rate(
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
    '--config-name', required=True, type=str, default='1-config-static-audio')
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
