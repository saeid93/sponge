"""
Iterate through all possible combination
of models and servers
"""

from email.policy import default
from importlib.metadata import requires
from operator import mod
import os
from pickletools import read_uint1
from platform import node
import re
import time
import json
import yaml
import click

from typing import Any, Dict
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import subprocess
from prom import get_cpu_usage, get_memory_usage
from datasets import load_dataset
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from barazmoon import MLServerBarAzmoon



# TODO from constants
pipelines_path = "/home/cc/infernece-pipeline-joint-optimization/pipelines/mlserver-prototype"
results_path = "/home/cc/infernece-pipeline-joint-optimization/data/nodes"
configs_path = "/home/cc/infernece-pipeline-joint-optimization/data/configs/profiling/single-node"
timeout = 5
# # TODO from click variables
# pipeline_name = "paper-audio-qa"
# node_name = "audio"
# config_name = ""

# TODO from config file
# config = {
#     "model_vairants" : ["facebook/s2t-small-librispeech-asr"],
#     "max_batch_size": ["5", "10"],
#     "max_batch_time": ["1", "10"],
#     "cpu_request": ["4"],
#     "memory_request": ["4Gi"],
#     "replicas": [2],
#     "data_type": "audio"
# }


# input_sample_path = os.path.join(
#     node_path, 'input-sample.json'
# )

def experiments(node_name: str, config: dict, node_path: str, experiment_type: str, data_type):
    model_vairants = config['model_vairants']
    max_batch_sizes = config['max_batch_size']
    max_batch_times = config['max_batch_time']
    cpu_requests = config['cpu_request']
    memory_requests = config["memory_request"]
    replicas = config['replicas']
    # Better solution instead of nested for loops
    for model_variant in model_vairants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replicas:
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
                            results = load_test(
                                node_name=node_name,
                                data_type=data_type,
                                node_path=node_path,
                                experiment_type=experiment_type)
                            time.sleep(timeout) # TODO better validation -> some request
                            # TODO remove node upon finishing the tests
                            # TODO add some sleep
                            remove_node(node_name=node_name)
                            save_report(results=results)

def setup_node(node_name: str, cpu_request: str, memory_request: str,
               model_variant: str, max_batch_size: str,
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
              node_path: str, experiment_type: str):
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
    # TODO load workload
    gateway_endpoint = "localhost:32000"
    namespace = "default"
    if experiment_type == 'static':
        workload = [10, 4, 8] # TODO fix
    elif experiment_type == 'dynamic':
        workload = [10, 4, 8] # TODO fix
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")
    endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{node_name}/v2/models/infer"
    load_tester = MLServerBarAzmoon(
        endpoint=endpoint,
        http_method='post',
        workload=workload,
        data=data,
        data_shape=data_shape,
        data_type=data_type)
    load_tester.start()
    a = 1
    pass


def remove_node(node_name):
    os.system(f"kubectl delete seldondeployment {node_name} -n default")

def save_report(results: dict):
    pass

@click.command()
@click.option('--pipeline-name', required=True, type=str, default='paper-audio-qa')
@click.option('--node-name', required=True, type=str, default='audio')
@click.option('--data-type', required=True, type=str, default='audio')
@click.option('--experiment-type', required=True, type=str, default='static')
@click.option('--config-name', required=True, type=str, default='config_1')
def main(pipeline_name: str, node_name: str, data_type: str,
         experiment_type: str, config_name: str):
    config_path = os.path.join(configs_path, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    node_path = os.path.join(
        pipelines_path,
        pipeline_name,
        'seldon-core-version',
        'nodes',
        node_name
    )
    experiments(
        node_name=node_name,
        config=config,
        node_path=node_path,
        experiment_type=experiment_type,
        data_type=data_type
        )

if __name__ == "__main__":
    main()
# -=================

PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-audio-qa/seldon-core-version"
PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline-test-meta" # TODO fix be moved to utilspr
DATABASE = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline"
CHECK_TIMEOUT = 60
RETRY_TIMEOUT = 90
DELETE_WAIT = 45
LOAD_TEST_WAIT = 60
TRIAL_END_WAIT = 60
TEMPLATE = "audio"
CONFIG_FILE = "paper-audio-qa"
save_path = os.path.join(DATABASE, "audio-qa-data-new")
if not os.path.exists(save_path):
    os.makedirs(save_path)


ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

inputs=ds[0]["audio"]["array"]

def change_names(names):
    return_names = []
    for name in names:
        return_names.append(name.replace("_", "-"))
    return return_names


def extract_node_timer(json_data : dict):
    keys = list(json_data.keys())
    nodes = []
    sir_names = ["arrival_"]
    for name in sir_names:
        for key in keys:
            if name in key:
                nodes.append(key.replace(name, ""))

    return_nodes = change_names(nodes)
    return_timer = {}
    for node in nodes:
        return_timer[node] = json_data["serving_" + node] - json_data["arrival_" + node]
    e2e_lats = json_data[keys[-1]] - json_data[keys[0]]

    return return_nodes, return_timer, e2e_lats
    
def load_test(
    pipeline_name: str,
    inputs: Dict[str, Any],
    node_1_model, 
    node_2_model,
    n_items: int,
    n_iters = 40
    ):
    start = time.time()
    gateway_endpoint="localhost:32000"
    deployment_name = pipeline_name 
    namespace = "default"
    num_nodes = pipeline_name.split("-").__len__()
    e2e_lats = []
    node_latencies = [[] for _ in range(num_nodes)]
    cpu_usages = [[] for _ in range(num_nodes) ]
    memory_usages = [[] for _ in range(num_nodes) ]
    sc = SeldonClient(
        gateway_endpoint=gateway_endpoint,
        gateway="istio",
        transport="rest",
        deployment_name=deployment_name,
        namespace=namespace)

    time.sleep(CHECK_TIMEOUT)
    for iter in range(n_iters):
        response = sc.predict(
            data=inputs
        )

        if response.success:
            json_data_timer = response.response['jsonData']['time']
            return_nodes, return_timer, e2e_lat = extract_node_timer(json_data_timer)
            for i , name in enumerate(return_nodes):
                cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
                memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, 1))
                e2e_lats.append(e2e_lat)
            for i, time_ in enumerate(return_timer.keys()):
                node_latencies[i].append(return_timer[time_])

        else:
            pp.pprint(response.msg)
        print(iter)
    time.sleep(CHECK_TIMEOUT)
    total_time = int((time.time() - start)//60)
    for i , name in enumerate(return_nodes):
        cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
        memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, total_time, True))
    models = node_1_model + "*" + node_2_model + "*"
    with open(save_path+"/cpu.txt", "a") as cpu_file:
        cpu_file.write(f"usage of {models} {pipeline_name} is {cpu_usages} \n")

    with open(save_path+"/memory.txt", 'a') as memory_file:
        memory_file.write(f"usage of {models} {pipeline_name} is {memory_usages} \n")


    with open(save_path+"/node-latency.txt", "a") as infer:
        infer.write(f"lats of {models} {pipeline_name} is {node_latencies} \n")
    
    with open(save_path+"/ee.txt", "a") as s:
        s.write(f"eelat of {models} {pipeline_name} is {e2e_lats} \n")
    
def setup_pipeline(
    node_1_model: str,
    node_2_model: str, 
    template: str,
    pipeline_name: str):
    svc_vars = {
        "node_1_variant": node_1_model,
        "node_2_variant": node_2_model,        
        "pipeline_name": pipeline_name}
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template(f"{template}.yaml")
    content = svc_template.render(svc_vars)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
    """
    os.system(command)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")

config_file_path = os.path.join(
    PIPELINES_MODELS_PATH, f"{CONFIG_FILE}.yaml")
with open(config_file_path, 'r') as cf:
    config = yaml.safe_load(cf)

node_1_models = config['node_1']
node_2_models = config['node_2']

def prune_name(name, len):
    forbidden_strs = ['facebook', '/', 'deepset', '-']
    for forbidden_str in forbidden_strs:
        name = name.replace(forbidden_str, '')
    name = name.lower()
    name = name[:len]
    return name

for node_1_model in node_1_models:
    for node_2_model in node_2_models:
        pipeline_name = prune_name(node_1_model, 8) + "-" +\
            prune_name(node_2_model, 8)
        start_time = time.time()
        while True:
            setup_pipeline(
                node_1_model=node_1_model,
                node_2_model=node_2_model,
                template=TEMPLATE, pipeline_name=pipeline_name)
            time.sleep(CHECK_TIMEOUT)
            command = ("kubectl rollout status deploy/$(kubectl get deploy"
                    f" -l seldon-deployment-id={pipeline_name} -o"
                    " jsonpath='{.items[0].metadata.name}')")
            time.sleep(CHECK_TIMEOUT)
            p = subprocess.Popen(command, shell=True)
            try:
                p.wait(RETRY_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                p.kill()
                print("corrupted pipeline, should be deleted ...")
                remove_pipeline(pipeline_name=pipeline_name)
                print('waiting to delete ...')
                time.sleep(DELETE_WAIT)

        print('starting the load test ...\n')
        load_test(pipeline_name=pipeline_name, inputs=inputs, node_1_model=node_1_model, node_2_model=node_2_model, n_items=1)

        time.sleep(DELETE_WAIT)

        print("operation done, deleting the pipeline ...")
        remove_pipeline(pipeline_name=pipeline_name)
        print('pipeline successfuly deleted')
