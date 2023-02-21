"""
Iterate through all possible combination
of models and servers
"""
import os
import time
import grpc
import json
import yaml
import click
import sys
import numpy as np
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
from experiments.utils.prometheus import SingleNodePromClient
# import experiments.utils.constants import
from experiments.utils.constants import (
    PIPLINES_PATH,
    NODE_PROFILING_CONFIGS_PATH,
    NODE_PROFILING_RESULTS_STATIC_PATH,
    OBJ_NODE_PROFILING_RESULTS_STATIC_PATH
)
from experiments.utils.obj import setup_obj_store
prom_client = SingleNodePromClient()
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

def experiments(pipeline_name: str, node_name: str,
                config: dict, node_path: str, data_type: str):
    model_variants = config['model_variants']
    max_batch_sizes = config['max_batch_size']
    max_batch_times = config['max_batch_time']
    cpu_requests = config['cpu_request']
    memory_requests = config["memory_request"]
    replicas = config['replicas']
    workload_config = config['workload_config']
    repetition = config['repetition']
    series = config['series']
    series_meta = config['series_meta']
    loads_to_test = workload_config['loads_to_test']
    load_duration = workload_config['load_duration']
    mode = config['mode']
    benchmark_duration = config['benchmark_duration']
    use_threading = config['use_threading']
    # for now set to the number of CPU
    num_interop_threads = config['num_interop_threads']
    num_threads = config['num_threads']
    if 'no_engine' in config.keys():
        no_engine = config['no_engine']
    else:
        no_engine = False
    timeout = config['timeout']
    # TOOD Add cpu type, gpu type
    # TODO Better solution instead of nested for loops
    # TODO Also add the random - maybe just use Tune
    for model_variant in model_variants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replicas:
                            # for num_interop_thread in num_interop_threads:
                            #     for num_thread in num_threads:
                            for load in loads_to_test:
                                # for rep in range(repetition):
                                print('-'*25\
                                    + f' starting repetition experiment ' +\
                                        '-'*25)
                                print('\n')
                                experiments_exist, experiment_id = key_config_mapper(
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
                                    replica=replica,
                                    no_engine=no_engine,
                                    mode=mode,
                                    data_type=data_type,
                                    benchmark_duration=benchmark_duration)
                                if not experiments_exist:
                                    setup_node(
                                        node_name=node_name,
                                        cpu_request=cpu_request,
                                        memory_request=memory_request,
                                        model_variant=model_variant,
                                        max_batch_size=max_batch_size,
                                        max_batch_time=max_batch_time,
                                        replica=replica,
                                        node_path=node_path,
                                        timeout=timeout,
                                        no_engine=no_engine,
                                        use_threading=use_threading,
                                        # HACK for now we set the number of requests
                                        # proportional to the the number threads
                                        num_interop_threads=cpu_request,
                                        num_threads=cpu_request
                                    )
                                    check_load_test(
                                        node_name=node_name, data_type=data_type,
                                                node_path=node_path)

                                    print('-'*25 + f' starting load test ' + '-'*25)
                                    print('\n')

                                    start_time_experiment,\
                                        end_time_experiment, responses = load_test(
                                            node_name=node_name,
                                            data_type=data_type,
                                            node_path=node_path,
                                            load=load,
                                            mode=mode,
                                            namespace='default',
                                            load_duration=load_duration,
                                            no_engine=no_engine,
                                            benchmark_duration=benchmark_duration)
                                    # TODO id system for the experiments
                                    save_report(
                                        experiment_id=experiment_id,
                                        responses = responses,
                                        node_name=node_name,
                                        start_time_experiment=start_time_experiment,
                                        end_time_experiment=end_time_experiment,
                                        series=series,
                                        no_engine=no_engine)
                                    
                                            # backup(series=series)

                                    # TODO better validation -> some request
                                    print(f'waiting for timeout: {timeout} seconds')
                                    for _ in tqdm(range(20)):
                                        time.sleep((timeout)/20)
                                    remove_node(node_name=node_name)
                                else:
                                    print('experiment with the same set of varialbes already exists')
                                    print('skipping to the next experiment ...')
                                    continue

def key_config_mapper(
    pipeline_name: str, node_name: str, cpu_request: str,
    memory_request: str, model_variant: str, max_batch_size: str,
    max_batch_time: str, load: int,
    load_duration: int, series: int, series_meta: str, replica: int,
    no_engine: bool = True, mode: str = 'step', data_type: str = 'audio',
    benchmark_duration=1):
    dir_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH,
        'series', str(series))
    file_path =  os.path.join(dir_path, KEY_CONFIG_FILENAME)
    header = [
        'experiment_id','pipeline_name', 'node_name',
        'model_variant', 'cpu_request',
        'memory_request', 'max_batch_size',
        'max_batch_time', 'load', 'load_duration',
        'series', 'series_meta', 'replicas', 'no_engine',
        'mode', 'data_type', 'benchmark_duration']
    row_to_add = {
        'experiment_id': None,
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
        'replicas': replica,
        'no_engine': no_engine,
        'mode': mode,
        'data_type': data_type,
        'benchmark_duration': benchmark_duration
        }
    experiments_exist = False
    if not os.path.exists(file_path):
        # os.makedirs(dir_path)
        with open(file_path, 'w', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
        experiment_id = 1
    else:
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                # add logic of experiment exists
                file_row_dict = {}
                if line_count != 0:
                    for key, value in zip(header, row):
                        if key in [
                            'pipeline_name', 'node_name',
                            'max_batch_size', 'max_batch_time',
                            'memory_request', 'model_variant',
                            'memory_request', 'cpu_request',
                            'series_meta', 'mode', 'data_type']:
                            file_row_dict[key] = value
                        elif key in [
                            'experiment_id', 'load',
                            'load_duration', 'series',
                            'replicas', 'benchmark_duration']:
                            file_row_dict[key] = int(value)
                        elif key in ['no_engine']:
                            file_row_dict[key] = eval(value)
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
                label_selector=f"seldon-deployment-id={node_name}-default")
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

def load_test(node_name: str, data_type: str,
              node_path: str,
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
            data = openfile.read()
        # with open(input_sample_shape_path, 'r') as openfile:
        #     data_shape = json.load(openfile)
        #     data_shape = data_shape['data_shape']
            data_shape = [1]
    elif data_type == 'image':
        input_sample_path = os.path.join(
            node_path, 'input-sample.JPEG'
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
    deployment_name = node_name
    model = node_name
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

def check_load_test(node_name: str, data_type: str,
              node_path: str,
              load=1, load_duration = 1):
    loop_timeout = 5
    ready = False
    while True:
        print(f'waited for {loop_timeout} to check for successful request')
        time.sleep(loop_timeout)
        try:
            load_test(
                node_name=node_name,
                data_type=data_type,
                node_path=node_path,
                load=load,
                load_duration=load_duration)
            ready = True
        except UnboundLocalError:
            pass
        if ready:
            return ready

def remove_node(node_name):
    os.system(f"kubectl delete seldondeployment {node_name} -n default")
    print('-'*50 + f' model pod {node_name} successfuly set up ' + '-'*50)
    print('\n')

def save_report(experiment_id: int,
                responses: str,
                node_name: str,
                start_time_experiment: float,
                end_time_experiment: float,
                namespace: str = 'default',
                series: int = 0,
                no_engine: bool = False,
                ):
    results = {
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
        'responses': responses,
        'start_time_experiment': start_time_experiment,
        'end_time_experiment': end_time_experiment,
    }
    # TODO experiments id system
    duration = (end_time_experiment - start_time_experiment)//60 + 1

    save_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH,
        'series', str(series), f"{experiment_id}.json")
    # TODO add list of pods in case of replicas
    pod_name = get_pod_name(node_name=node_name)[0]

    if not no_engine:
        svc_path = os.path.join(
            NODE_PROFILING_RESULTS_STATIC_PATH,
            'series', str(series), f"{experiment_id}.txt")
        svc_pod_name = get_pod_name(
            node_name=node_name, orchestrator=True)
    cpu_usage_count, time_cpu_usage_count =\
        prom_client.get_cpu_usage_count(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name)
    cpu_usage_rate, time_cpu_usage_rate =\
        prom_client.get_cpu_usage_rate(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name, rate=120)

    cpu_throttled_count, time_cpu_throttled_count =\
        prom_client.get_cpu_throttled_count(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name)
    cpu_throttled_rate, time_cpu_throttled_rate =\
        prom_client.get_cpu_throttled_rate(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name, rate=120)

    memory_usage, time_memory_usage = prom_client.get_memory_usage(
        pod_name=pod_name, namespace="default",
        container=node_name, duration=int(duration), need_max=False)

    throughput, time_throughput = prom_client.get_request_per_second(
            pod_name=pod_name, namespace="default",
            duration=int(duration), container=node_name, rate=120)

    results['cpu_usage_count'] = cpu_usage_count
    results['time_cpu_usage_count'] = time_cpu_usage_count
    results['cpu_usage_rate'] = cpu_usage_rate
    results['time_cpu_usage_rate'] = time_cpu_usage_rate

    results['cpu_throttled_count'] = cpu_throttled_count
    results['time_cpu_throttled_count'] = time_cpu_throttled_count
    results['cpu_throttled_rate'] = cpu_throttled_rate
    results['time_cpu_throttled_rate'] = time_cpu_throttled_rate

    results['memory_usage'] = memory_usage
    results['time_memory_usage'] = time_memory_usage

    results['throughput'] = throughput
    results['time_throughput'] = time_throughput

    with open(save_path, "w") as outfile:
        outfile.write(json.dumps(results))

    if not no_engine:
        os.system(
            f'kubectl logs -n {namespace} {svc_pod_name} > {svc_path}'
        )
    print(f'results have been sucessfully saved in:\n{save_path}')

def backup(series):
    data_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH,
        'series', str(series))
    backup_path = os.path.join(
        OBJ_NODE_PROFILING_RESULTS_STATIC_PATH,
        'series', str(series))
    setup_obj_store()
    shutil.copytree(data_path, backup_path)

@click.command()
@click.option(
    '--config-name', required=True, type=str, default='5-config-static-resnet-human')
def main(config_name: str):
    config_path = os.path.join(
        NODE_PROFILING_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    pipeline_name = config['pipeline_name']
    node_name = config['node_name']
    data_type = config['data_type']
    series = config['series']
    node_path = os.path.join(
        PIPLINES_PATH,
        pipeline_name,
        'seldon-core-version',
        'nodes',
        node_name
    )

    dir_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH,
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
        node_name=node_name,
        config=config,
        node_path=node_path,
        data_type=data_type
        )

if __name__ == "__main__":
    main()
