"""
Iterate through all possible combination
of pipelines
"""
import os
import time
import json
import yaml
import click
import sys
import numpy as np
from typing import Tuple
import itertools
import csv
import re
import asyncio
from jinja2 import Environment, FileSystemLoader
from barazmoon import Data
from PIL import Image
from kubernetes import config
from kubernetes import client
# from kubernetes.client import Configuration
# from kubernetes.client.api import core_v1_api
from tqdm import tqdm
import shutil
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from barazmoon import (
    MLServerAsyncGrpc)

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..', '..', '..')))
from experiments.utils.prometheus import PromClient
# import experiments.utils.constants import
from experiments.utils.constants import (
    PIPLINES_PATH,
    PIPELINE_PROFILING_CONFIGS_PATH,
    PIPELINE_PROFILING_RESULTS_PATH,
    OBJ_PIPELINE_PROFILING_RESULTS_PATH
)
from experiments.utils.obj import setup_obj_store
prom_client = PromClient()
try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)
kube_api = client.api.core_v1_api.CoreV1Api()

KEY_CONFIG_FILENAME = 'key_config_mapper.csv'
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

def experiments(pipeline_name: str, node_names: str,
                config: dict, pipeline_path: str,
                data_type: str):
    workload_config = config['workload_config']
    repetition = config['repetition']
    series = config['series']
    metadata = config['metadata']
    loads_to_test = workload_config['loads_to_test']
    load_duration = workload_config['load_duration']
    timeout = config['timeout']
    mode = config['mode']
    benchmark_duration = config['benchmark_duration']

    model_variants = []
    max_batch_sizes = []
    max_batch_times = []
    cpu_requests = []
    memory_requests = []
    replicas = []
    use_threading = []
    num_iterop_threads = []
    num_threads = []
    for node_config in config['nodes']:
        model_variants.append(node_config['model_variants'])
        max_batch_sizes.append(node_config['max_batch_size'])
        max_batch_times.append(node_config['max_batch_time'])
        cpu_requests.append(node_config['cpu_request'])
        memory_requests.append(node_config["memory_request"])
        replicas.append(node_config['replicas'])
        use_threading.append([node_config['use_threading']])
        num_iterop_threads.append(node_config['num_interop_threads'])
        num_threads.append(node_config['num_threads'])


    model_variants = list(itertools.product(*model_variants))
    max_batch_sizes = list(itertools.product(*max_batch_sizes))
    max_batch_times = list(itertools.product(*max_batch_times))
    cpu_requests = list(itertools.product(*cpu_requests))
    memory_requests = list(itertools.product(*memory_requests))
    replicas = list(itertools.product(*replicas))
    use_threading = list(itertools.product(*use_threading))[0]
    num_iterop_threads = list(itertools.product(*num_iterop_threads))
    num_threads = list(itertools.product(*num_threads))
    # TOOD Add cpu type, gpu type
    # TODO Better solution instead of nested for loops
    # TODO Also add the random - maybe just use Tune
    for model_variant in model_variants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replicas:
                            for load in loads_to_test:
                                print('-'*25\
                                    + f' starting repetition experiment ' +\
                                        '-'*25)
                                print('\n')
                                experiments_exist, experiment_id = key_config_mapper(
                                    pipeline_name=pipeline_name,
                                    node_name=node_names,
                                    cpu_request=cpu_request,
                                    memory_request=memory_request,
                                    model_variant=model_variant,
                                    max_batch_size=max_batch_size,
                                    max_batch_time=max_batch_time,
                                    load=load,
                                    load_duration=load_duration,
                                    series=series,
                                    metadata=metadata,
                                    replica=replica,
                                    mode=mode,
                                    data_type=data_type,
                                    benchmark_duration=benchmark_duration)
                                if not experiments_exist:
                                    setup_pipeline(
                                        # pipeline_folder_name=pipeline_folder_name,
                                        pipeline_name=pipeline_name,
                                        cpu_request=cpu_request,
                                        memory_request=memory_request,
                                        model_variant=model_variant,
                                        max_batch_size=max_batch_size,
                                        max_batch_time=max_batch_time,
                                        replica=replica,
                                        pipeline_path=pipeline_path,
                                        timeout=timeout,
                                        num_nodes=len(config['nodes']),
                                        use_threading=use_threading,
                                        # HACK for now we set the number of requests
                                        # proportional to the the number threads
                                        num_interop_threads=cpu_request,
                                        num_threads=cpu_request
                                    )

                                    print('Checking if the model is up ...')
                                    print('\n')
                                    # check if the model is up or not
                                    check_load_test(
                                        pipeline_name=pipeline_name,
                                        data_type=data_type,
                                        pipeline_path=pipeline_path)
                                    print('model warm up ...')
                                    print('\n')
                                    warm_up_duration = 10
                                    warm_up(
                                        pipeline_name=pipeline_name,
                                        data_type=data_type,
                                        pipeline_path=pipeline_path,
                                        load_duration=warm_up_duration)
                                    print('-'*25 + f'starting load test ' + '-'*25)
                                    print('\n')
                                    print('-'*25 + f'starting load test ' + '-'*25)
                                    print('\n')
                                    try:
                                        start_time_experiment,\
                                            end_time_experiment, responses = load_test(
                                                pipeline_name=pipeline_name,
                                                data_type=data_type,
                                                pipeline_path=pipeline_path,
                                                load=load,
                                                mode=mode,
                                                namespace='default',
                                                load_duration=load_duration,
                                                benchmark_duration=benchmark_duration)
                                        print('-'*25 + 'saving the report' + '-'*25)
                                        print('\n')
                                        save_report(
                                            experiment_id=experiment_id,
                                            responses=responses,
                                            pipeline_name=pipeline_name,
                                            node_names=node_names,
                                            start_time_experiment=start_time_experiment,
                                            end_time_experiment=end_time_experiment,
                                            series=series)
                                    except UnboundLocalError:
                                        print('Impossible experiment!')
                                        print('skipping to the next experiment ...')
                                    print(f'waiting for timeout: {timeout} seconds')
                                    for _ in tqdm(range(20)):
                                        time.sleep((timeout)/20)
                                    remove_pipeline(pipeline_name=pipeline_name)
                                else:
                                    print('experiment with the same set of varialbes already exists')
                                    print('skipping to the next experiment ...')
                                    continue

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

def key_config_mapper(
    pipeline_name: str, node_name: Tuple[str], cpu_request: Tuple[str],
    memory_request: Tuple[str], model_variant: Tuple[str],
    max_batch_size: Tuple[str], max_batch_time: Tuple[str], load: int,
    load_duration: int, series: int, metadata: str, replica: int,
    mode: str = 'step', data_type: str = 'audio',
    benchmark_duration=1):
    dir_path = os.path.join(
        PIPELINE_PROFILING_RESULTS_PATH,
        'series', str(series))
    file_path =  os.path.join(dir_path, KEY_CONFIG_FILENAME)
    # header = [
    #     'experiment_id','pipeline_name', 'node_name',
    #     'model_variant', 'cpu_request',
    #     'memory_request', 'max_batch_size',
    #     'max_batch_time', 'load', 'load_duration',
    #     'series', 'metadata', 'replicas', 'no_engine',
    #     'mode', 'data_type', 'benchmark_duration']
    header = [
        'experiment_id','pipeline_name', 'load',
        'load_duration', 'series', 'metadata',
        'mode', 'data_type',
        'benchmark_duration']
    for node_index in range(len(node_name)):
        header += [f'task_{node_index}_node_name']
        header += [f'task_{node_index}_model_variant']
        header += [f'task_{node_index}_cpu_request']
        header += [f'task_{node_index}_memory_request']
        header += [f'task_{node_index}_max_batch_size']
        header += [f'task_{node_index}_max_batch_time']
        header += [f'task_{node_index}_replica']
    row_to_add = {
        'experiment_id': None,
        'pipeline_name': pipeline_name,
        # 'model_variant': model_variant,
        # 'node_name': node_name,
        # 'cpu_request': cpu_request,
        # 'memory_request': memory_request,
        # 'max_batch_size': max_batch_size,
        # 'max_batch_time': max_batch_time,
        # 'replicas': replica,
        'load': load,
        'load_duration': load_duration,
        'series': series,
        'metadata': metadata,
        'mode': mode,
        'data_type': data_type,
        'benchmark_duration': benchmark_duration
        }
    for node_index in range(len(node_name)):
        row_to_add[f'task_{node_index}_node_name'] = node_name[node_index]
        row_to_add[f'task_{node_index}_model_variant'] = model_variant[node_index]
        row_to_add[f'task_{node_index}_cpu_request'] = cpu_request[node_index]
        row_to_add[f'task_{node_index}_memory_request'] = memory_request[node_index]
        row_to_add[f'task_{node_index}_max_batch_size'] = max_batch_size[node_index]
        row_to_add[f'task_{node_index}_max_batch_time'] = max_batch_time[node_index]
        row_to_add[f'task_{node_index}_replica'] = replica[node_index]
    experiments_exist = False
    if not os.path.exists(file_path):
        # os.makedirs(dir_path)
        with open(file_path, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
        experiment_id = 1
    else:
        # write to the file if the row does not exist
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                # add logic of experiment exists
                file_row_dict = {}
                if line_count != 0:
                    for key, value in zip(header, row):
                        strings = [
                            'pipeline_name', 'node_name',
                            'max_batch_size', 'max_batch_time',
                            'memory_request', 'model_variant',
                            'memory_request', 'cpu_request',
                            'metadata', 'mode', 'data_type']
                        for string in strings:
                            if string in key:
                                file_row_dict[key] = value
                                break
                        integers = [
                            'experiment_id', 'load',
                            'load_duration', 'series',
                            'replica', 'benchmark_duration']
                        for integer in integers:
                            if integer in key:
                                file_row_dict[key] = int(value)
                                break
                    dict_items_equal = []
                    for header_item in header:
                        if header_item == 'experiment_id':
                            continue
                        if row_to_add[header_item] ==\
                            file_row_dict[header_item]:
                            dict_items_equal.append(True)
                        else:
                            dict_items_equal.append(False)
                    if all(dict_items_equal):
                        experiments_exist = True
                        break
                line_count += 1
        experiment_id = line_count

    if not experiments_exist:
        row_to_add.update({'experiment_id': experiment_id})
        with open(file_path, 'a') as row_writer:
            dictwriter_object = csv.DictWriter(row_writer, fieldnames=header)
            dictwriter_object.writerow(row_to_add)
            row_writer.close()

    return experiments_exist, experiment_id

def load_test(pipeline_name: str, data_type: str,
              pipeline_path: str,
              load: int, load_duration: int,
              namespace: str='default',
              no_engine: bool = False,
              mode: str = 'step',
              benchmark_duration=1):
    start_time = time.time()
    # load sample data
    # TODO change here
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
    workload = [load] * load_duration

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

def check_load_test(
        pipeline_name: str, data_type: str,
        pipeline_path: str,
        load=1, load_duration = 1):
    loop_timeout = 5
    ready = False
    while True:
        print(f'waited for {loop_timeout} seconds to check for successful request')
        time.sleep(loop_timeout)
        try:
            load_test(
                pipeline_name=pipeline_name,
                data_type=data_type,
                pipeline_path=pipeline_path,
                load=load,
                load_duration=load_duration)
            ready = True
        except UnboundLocalError:
            pass
        if ready:
            return ready

def warm_up(
        pipeline_name: str, data_type: str,
        pipeline_path: str,
        load_duration: int, load=1):
    load_test(
        pipeline_name=pipeline_name,
        data_type=data_type,
        pipeline_path=pipeline_path,
        load=load,
        load_duration=load_duration)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")
    print('-'*50 + f' model pod {pipeline_name} successfuly set up ' + '-'*50)
    print('\n')

def save_report(experiment_id: int,
                responses: str,
                pipeline_name: str,
                node_names: Tuple[str],
                start_time_experiment: float,
                end_time_experiment: float,
                namespace: str = 'default',
                series: int = 0,
                replicas: Tuple[int]= (1,1)):
    results = {
        'responses': responses,
        'start_time_experiment': start_time_experiment,
        'end_time_experiment': end_time_experiment,
    }
    # TODO add per pipeline id
    save_path = os.path.join(
        PIPELINE_PROFILING_RESULTS_PATH,
        'series', str(series), f"{experiment_id}.json")
    rate = int(end_time_experiment - start_time_experiment)
    duration = (end_time_experiment - start_time_experiment)//60 + 1
    pod_names = get_pod_name(node_name=pipeline_name)
    for node_name in node_names:
        node_pod_names = [s for s in pod_names if node_name in s]
        node_results = {}
        for node_pod_name in node_pod_names:
            pod_results = {
                'cpu_usage_count': [],
                'time_cpu_usage_count': [],
                'cpu_usage_rate': [],
                'time_cpu_usage_rate': [],
                'cpu_throttled_count': [],
                'time_cpu_throttled_count': [],
                'cpu_throttled_rate': [],
                'time_cpu_throttled_rate': [],
                'memory_usage': [],
                'time_memory_usage': [],
                'throughput': [],
                'time_throughput': [],
            }

            svc_path = os.path.join(
                PIPELINE_PROFILING_RESULTS_PATH,
                'series', str(series), f"{experiment_id}.txt")
            svc_pod_name = get_pod_name(
                node_name=node_name, orchestrator=True)
            cpu_usage_count, time_cpu_usage_count =\
                prom_client.get_cpu_usage_count(
                    pod_name=node_pod_name, namespace="default",
                    duration=int(duration), container=node_name)
            cpu_usage_rate, time_cpu_usage_rate =\
                prom_client.get_cpu_usage_rate(
                    pod_name=node_pod_name, namespace="default",
                    duration=int(duration), container=node_name, rate=rate)

            cpu_throttled_count, time_cpu_throttled_count =\
                prom_client.get_cpu_throttled_count(
                    pod_name=node_pod_name, namespace="default",
                    duration=int(duration), container=node_name)
            cpu_throttled_rate, time_cpu_throttled_rate =\
                prom_client.get_cpu_throttled_rate(
                    pod_name=node_pod_name, namespace="default",
                    duration=int(duration), container=node_name, rate=rate)

            memory_usage, time_memory_usage = prom_client.get_memory_usage(
                pod_name=node_pod_name, namespace="default",
                container=node_name, duration=int(duration), need_max=False)

            throughput, time_throughput = prom_client.get_request_per_second(
                    pod_name=node_pod_name, namespace="default",
                    duration=int(duration), container=node_name, rate=rate)

            pod_results['cpu_usage_count'] = cpu_usage_count
            pod_results['time_cpu_usage_count'] = time_cpu_usage_count
            pod_results['cpu_usage_rate'] = cpu_usage_rate
            pod_results['time_cpu_usage_rate'] = time_cpu_usage_rate

            pod_results['cpu_throttled_count'] = cpu_throttled_count
            pod_results['time_cpu_throttled_count'] = time_cpu_throttled_count
            pod_results['cpu_throttled_rate'] = cpu_throttled_rate
            pod_results['time_cpu_throttled_rate'] = time_cpu_throttled_rate

            pod_results['memory_usage'] = memory_usage
            pod_results['time_memory_usage'] = time_memory_usage

            pod_results['throughput'] = throughput
            pod_results['time_throughput'] = time_throughput

            node_results[node_pod_name] = pod_results

        # consider replicas
        results[node_name] = node_results

    with open(save_path, "w") as outfile:
        outfile.write(json.dumps(results))
    os.system(
        f'kubectl logs -n {namespace} {svc_pod_name} > {svc_path}'
    )
    print(f'results have been sucessfully saved in:\n{save_path}')

@click.command()
@click.option(
    '--config-name', required=True, type=str, default='video')
def main(config_name: str):
    config_path = os.path.join(
        PIPELINE_PROFILING_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    pipeline_name = config['pipeline_name']
    pipeline_folder_name = config['pipeline_folder_name']
    node_names = [config['node_name'] for config in config['nodes']]
    # first node of the pipeline determins the pipeline data_type
    data_type = config['nodes'][0]['data_type']
    series = config['series']
    pipeline_path = os.path.join(
        PIPLINES_PATH,
        pipeline_folder_name,
        'seldon-core-version'
    )

    dir_path = os.path.join(
        PIPELINE_PROFILING_RESULTS_PATH,
        'series', str(series))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        dest_config_path = os.path.join(
            dir_path,
            '0.yaml'
        )
        shutil.copy(config_path, dest_config_path)
    else:
        num_configs = 0
        # Iterate directory
        for file in os.listdir(dir_path):
            # check only text files
            if file.endswith('.yaml'):
                num_configs += 1
        dest_config_path = os.path.join(
            dir_path,
            f'{num_configs}.yaml'
        )
        shutil.copy(config_path, dest_config_path)

    experiments(
        pipeline_name=pipeline_name,
        node_names=node_names,
        config=config,
        pipeline_path=pipeline_path,
        data_type=data_type
        )

if __name__ == "__main__":
    main()
