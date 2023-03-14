import os
import time
import json
import yaml
import numpy as np
from typing import List, Tuple
import re
import numpy as np
from jinja2 import Environment, FileSystemLoader
from kubernetes import config
from kubernetes import client
from PIL import Image
import asyncio
from barazmoon import Data
from barazmoon import MLServerAsyncGrpc
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)
kube_api = client.api.core_v1_api.CoreV1Api()

NAMESPACE='default'

def get_pod_name(node_name: str, orchestrator=False):
    pod_regex = f"{node_name}.*"
    pods_list = kube_api.list_namespaced_pod(NAMESPACE)
    pod_names = []
    for pod_name in pods_list.items:
        pod_name=pod_name.metadata.name
        if orchestrator and re.match(pod_regex, pod_name) and 'svc' in pod_name:
            return pod_name
        if re.match(pod_regex, pod_name) and 'svc' not in pod_name:
            pod_names.append(pod_name)
    return pod_names

def setup_node(node_name: str, cpu_request: str,
               memory_request: str, model_variant: str, max_batch_size: str,
               max_batch_time: str, replica: int, node_path: str, timeout: int,
               use_threading: bool, num_interop_threads: int, num_threads: int,
               no_engine=False):
    print('-'*25 + ' setting up the node with following config' + '-'*25)
    print('\n')
    svc_vars = {
        "name": node_name,
        "cpu_request": cpu_request,
        "memory_request": memory_request,
        "cpu_limit": cpu_request,
        "memory_limit": memory_request,
        "model_variant": model_variant,
        "max_batch_size": max_batch_size,
        "max_batch_time": max_batch_time,
        "replicas": replica,
        "no_engine": str(no_engine),
        "use_threading": use_threading,
        "num_interop_threads": num_interop_threads,
        "num_threads": num_threads
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
    print('-'*25 + f' waiting to make sure the node is up ' + '-'*25)
    print('\n')
    print('-'*25 + f' model pod {node_name} successfuly set up ' + '-'*25)
    print('\n')
    # checks if the pods are ready each 5 seconds
    loop_timeout = 5
    while True:
        models_loaded, svc_loaded, container_loaded = False, False, False
        print(f'waited for {loop_timeout} to check if the pods are up')
        time.sleep(loop_timeout)
        model_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"seldon-deployment-id={node_name}")
        all_model_pods = []
        all_conainers = []
        for pod in model_pods.items:
            if pod.status.phase == "Running":
                all_model_pods.append(True)
                logs = kube_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=NAMESPACE,
                    container=node_name)
                print(logs)
                if 'Uvicorn running on http://0.0.0.0:6000' in logs:
                    all_conainers.append(True)
                else:
                    all_conainers.append(False)
            else:
                all_model_pods.append(False)
        print(f"all_model_pods: {all_model_pods}")
        if all(all_model_pods):
            models_loaded = True
        else: continue
        print(f"all_containers: {all_conainers}")
        if all(all_model_pods):
            container_loaded = True
        else: continue
        if not no_engine:
            svc_pods = kube_api.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"seldon-deployment-id={node_name}-{node_name}")
            for pod in svc_pods.items:
                if pod.status.phase == "Running":
                    svc_loaded = True
                for container_status in pod.status.container_statuses:
                    if container_status.ready:
                        container_loaded = True
                    else: continue
                else: continue
        if models_loaded and svc_loaded and container_loaded:
            print('model container completely loaded!')
            break


def setup_pipeline(pipeline_name: str,
                   cpu_request: Tuple[str], memory_request: Tuple[str],
                   model_variant: Tuple[str], max_batch_size: Tuple[str],
                   max_batch_time: Tuple[str], replica: Tuple[int],
                   use_threading: Tuple[bool], num_interop_threads: Tuple[int],
                   num_threads: Tuple[int], pipeline_path: str,
                   timeout: int, num_nodes: int):
    print('-'*25 + ' setting up the node with following config' + '-'*25)
    print('\n')
    # TODO add num nodes logic here
    svc_vars = {"name": pipeline_name}
    for node_id in range(num_nodes):
        node_index = node_id + 1
        svc_vars.update(
            {
                f"cpu_request_{node_index}": cpu_request[node_id],
                f"memory_request_{node_index}": memory_request[node_id],
                f"cpu_limit_{node_index}": cpu_request[node_id],
                f"memory_limit_{node_index}": memory_request[node_id],
                f"model_variant_{node_index}": model_variant[node_id],
                f"max_batch_size_{node_index}": max_batch_size[node_id],
                f"max_batch_time_{node_index}": max_batch_time[node_id],
                f"replicas_{node_index}": replica[node_id],
                f"use_threading_{node_index}": use_threading[node_id],
                f"num_interop_threads_{node_index}": num_interop_threads[node_id],
                f"num_threads_{node_index}": num_threads[node_id]
            })
    environment = Environment(
        loader=FileSystemLoader(pipeline_path))
    svc_template = environment.get_template('pipeline-template.yaml')
    content = svc_template.render(svc_vars)
    pp.pprint(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    print('-'*25 + f' waiting to make sure the node is up ' + '-'*25)
    print('\n')
    print('-'*25 + f' model pod {pipeline_name} successfuly set up ' + '-'*25)
    print('\n')
    # extract model model container names
    model_container = yaml.safe_load(content)
    model_names = list(
        map(
        lambda l: l['spec']['containers'][0]['name'],
        model_container['spec']['predictors'][0]['componentSpecs']))
    # checks if the pods are ready each 5 seconds
    loop_timeout = 5
    while True:
        models_loaded, svc_loaded, pipeline_loaded = False, False, False
        print(f'waited for {loop_timeout} to check if the pods are up')
        time.sleep(loop_timeout)
        model_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"seldon-deployment-id={pipeline_name}")
        all_model_pods = []
        all_conainers = []
        for pod in model_pods.items:
            if pod.status.phase == "Running":
                all_model_pods.append(True)
                pod_name = pod.metadata.name
                for model_name in model_names:
                    if model_name in pod_name:
                        container_name = model_name
                        break
                logs = kube_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=NAMESPACE,
                    container=container_name)
                print(logs)
                if 'Uvicorn running on http://0.0.0.0:600' in logs:
                    all_conainers.append(True)
                else:
                    all_conainers.append(False)
            else:
                all_model_pods.append(False)
        print(f"all_model_pods: {all_model_pods}")
        if all(all_model_pods):
            models_loaded = True
        else: continue
        print(f"all_containers: {all_conainers}")
        if all(all_model_pods):
            pipeline_loaded = True
        else: continue
        svc_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"seldon-deployment-id={pipeline_name}-{pipeline_name}")
        for pod in svc_pods.items:
            if pod.status.phase == "Running":
                svc_loaded = True
            for container_status in pod.status.container_statuses:
                if container_status.ready:
                    pipeline_loaded = True
                else: continue
            else: continue
        if models_loaded and svc_loaded and pipeline_loaded:
            print('model container completely loaded!')
            break

def load_data(data_type: str, pipeline_path: str):
    if data_type == 'audio':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.json'
        )
        input_sample_shape_path = os.path.join(
            pipeline_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = json.load(openfile)
        with open(input_sample_shape_path, 'r') as openfile:
            data_shape = json.load(openfile)
            data_shape = data_shape['data_shape']
    elif data_type == 'text':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.txt'
        )
        input_sample_shape_path = os.path.join(
            pipeline_path, 'input-sample-shape.json'
        )
        with open(input_sample_path, 'r') as openfile:
            data = openfile.read()
        # with open(input_sample_shape_path, 'r') as openfile:
        #     data_shape = json.load(openfile)
        #     data_shape = data_shape['data_shape']
            data_shape = [1]
    elif data_type == 'image':
        input_sample_path = os.path.join(
            pipeline_path, 'input-sample.JPEG'
        )
        data = Image.open(input_sample_path)
        data_shape = list(np.array(data).shape)
        data = np.array(data).flatten()
    data_1 = Data(
        data=data,
        data_shape=data_shape,
        custom_parameters={'custom': 'custom'},
    )

    # Data list
    data = []
    data.append(data_1)
    return data

def check_load_test(
        pipeline_name: str, data_type: str,
        pipeline_path: str):
    data = load_data(
        data_type=data_type,
        pipeline_path=pipeline_path)
    loop_timeout = 5
    ready = False
    while True:
        print(f'waited for {loop_timeout} seconds to check for successful request')
        time.sleep(loop_timeout)
        try:
            load_test(
                pipeline_name=pipeline_name,
                data=data,
                data_type=data_type,
                workload=[1])
            ready = True
        except UnboundLocalError:
            pass
        if ready:
            return ready

def warm_up(
        pipeline_name: str, data_type: str,
        pipeline_path: str, warm_up_duration: int):
    workload = [1] * warm_up_duration
    data = load_data(
        data_type=data_type,
        pipeline_path=pipeline_path)
    load_test(
        pipeline_name=pipeline_name,
        data=data,
        data_type=data_type,
        workload=workload)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")
    print('-'*50 + f' pipeline {pipeline_name} successfuly removed ' + '-'*50)
    print('\n')

def load_test(
        pipeline_name: str, data_type: str,
        workload: List[int],
        data: List[Data],
        namespace: str='default',
        no_engine: bool = False,
        mode: str = 'step',
        benchmark_duration=1):
    start_time = time.time()

    endpoint = "localhost:32000"
    deployment_name = pipeline_name
    model = None
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
    load_tester = MLServerAsyncGrpc(
        endpoint=endpoint,
        metadata=metadata,
        workload=workload,
        model=model,
        data=data,
        mode=mode, # options - step, equal, exponential
        data_type=data_type,
        benchmark_duration=benchmark_duration)
    responses = asyncio.run(load_tester.start())
    end_time = time.time()

    # remove ouput for image inputs/outpus (yolo)
    # as they make logs very heavy
    for second_response in responses:
        for response in second_response:
            response['outputs'] = []
    return start_time, end_time, responses