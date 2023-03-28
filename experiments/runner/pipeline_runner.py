"""
Iterate through all possible combination
of pipelines
"""
import os
import time
import json
import yaml
from typing import Union, Tuple
import click
import sys
import csv
from tqdm import tqdm
import shutil
from barazmoon.twitter import twitter_workload_generator
import multiprocessing

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..', '..')))
from experiments.utils.prometheus import PromClient
from experiments.utils.pipeline_operations import (
    load_data,
    warm_up,
    check_load_test,
    load_test,
    remove_pipeline,
    setup_router_pipeline,
    get_pod_name)
from experiments.utils.constants import (
    PIPLINES_PATH,
    FINAL_CONFIGS_PATH,
    FINAL_RESULTS_PATH,
    ACCURACIES_PATH,
    OBJ_FINAL_RESULTS_PATH,
    KEY_CONFIG_FILENAME
)
from experiments.utils.obj import setup_obj_store
prom_client = PromClient()
from optimizer import (
    Optimizer,
    Adapter
    )
from experiments.utils.simulation_operations import (
    generate_simulated_pipeline
)

def setup_pipeline(
        pipeline_name: str, node_names: str,
        config: dict, pipeline_path: str,
        data_type: str):

    series = config['series']
    metadata = config['metadata']
    timeout = config['timeout']
    mode = config['mode']
    benchmark_duration = config['benchmark_duration']
    workload_type = config['workload_type']
    workload_config = config['workload_config']
    if workload_type == 'static':
        loads_to_test = workload_config['loads_to_test']
        load_duration = workload_config['load_duration']
    elif workload_type == 'twitter':
        # loads_to_test = workload_config['loads_to_test']
        loads_to_test = []
        for w_config in workload_config:
            start = w_config['start']
            end = w_config['end']
            load_to_test = start + '-' + end
            loads_to_test.append(load_to_test)
        workload = twitter_workload_generator(loads_to_test[0])
        load_duration = len(workload)

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
        use_threading.append([node_config['use_threading']][0])
        num_iterop_threads.append(node_config['num_interop_threads'])
        num_threads.append(node_config['num_threads'])

    experiments_exist, experiment_id = key_config_mapper(
        pipeline_name=pipeline_name,
        node_name=node_names,
        cpu_request=cpu_requests,
        memory_request=memory_requests,
        model_variant=model_variants,
        max_batch_size=max_batch_sizes,
        max_batch_time=max_batch_times,
        load=loads_to_test,
        load_duration=load_duration,
        series=series,
        metadata=metadata,
        replica=replicas,
        mode=mode,
        data_type=data_type,
        benchmark_duration=benchmark_duration)

    setup_router_pipeline(
        node_names=node_names,
        pipeline_name=pipeline_name,
        cpu_request=cpu_requests,
        memory_request=memory_requests,
        model_variant=model_variants,
        max_batch_size=max_batch_sizes,
        max_batch_time=max_batch_times,
        replica=replicas,
        pipeline_path=pipeline_path,
        timeout=timeout,
        num_nodes=len(config['nodes']),
        use_threading=use_threading,
        # HACK for now we set the number of requests
        # proportional to the the number threads
        num_interop_threads=cpu_requests,
        num_threads=cpu_requests
    )
    print('Checking if the model is up ...')
    print('\n')
    # check if the model is up or not
    check_load_test(
        pipeline_name='router',
        model='router',
        data_type=data_type,
        pipeline_path=pipeline_path)
    print('model warm up ...')
    print('\n')
    warm_up_duration = 10
    warm_up(
        pipeline_name='router',
        model='router',
        data_type=data_type,
        pipeline_path=pipeline_path,
        warm_up_duration=warm_up_duration)

    print('-'*25 + f'starting load test ' + '-'*25)
    print('\n')
    print('-'*25 + f'starting load test ' + '-'*25)
    print('\n')
    return experiments_exist, experiment_id

def experiments(
        pipeline_name: str, node_names: str,
        config: dict, pipeline_path: str,
        data_type: str, experiment_id: int):

    series = config['series']
    mode = config['mode']
    benchmark_duration = config['benchmark_duration']
    workload_type = config['workload_type']
    workload_config = config['workload_config']
    if workload_type == 'static':
        loads_to_test = workload_config['loads_to_test']
        load_duration = workload_config['load_duration']
    elif workload_type == 'twitter':
        # loads_to_test = workload_config['loads_to_test']
        loads_to_test = []
        for w_config in workload_config:
            start = w_config['start']
            end = w_config['end']
            load_to_test = start + '-' + end
            loads_to_test.append(load_to_test)
        workload = twitter_workload_generator(loads_to_test[0])
        load_duration = len(workload)

    if workload_type == 'static':
        workload = [loads_to_test] * load_duration
    data = load_data(data_type, pipeline_path)
    try:
        start_time_experiment,\
            end_time_experiment, responses = load_test(
                pipeline_name='router',
                model='router',
                data_type=data_type,
                data=data,
                workload=workload,
                mode=mode,
                namespace='default',
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

def key_config_mapper(
    pipeline_name: str, node_name: Tuple[str], cpu_request: Tuple[str],
    memory_request: Tuple[str], model_variant: Tuple[str],
    max_batch_size: Tuple[str], max_batch_time: Tuple[str],
    load: Union[int, str], load_duration: int, series: int, metadata: str,
    replica: int, mode: str = 'step', data_type: str = 'audio',
    benchmark_duration=1):
    dir_path = os.path.join(
        FINAL_RESULTS_PATH,
        'series', str(series))
    file_path =  os.path.join(dir_path, KEY_CONFIG_FILENAME)
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
                                try:
                                    file_row_dict[key] = int(value)
                                except ValueError:
                                    # for twitter loads
                                    file_row_dict[key] = value
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
        FINAL_RESULTS_PATH,
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
    print(f'results have been sucessfully saved in:\n{save_path}')

def backup(series):
    data_path = os.path.join(
        FINAL_RESULTS_PATH,
        'series', str(series))
    backup_path = os.path.join(
        FINAL_RESULTS_PATH,
        'series', str(series))
    setup_obj_store()
    shutil.copytree(data_path, backup_path)

@click.command()
@click.option(
    '--config-name', required=True, type=str, default='video')
def main(config_name: str):
    """loading system configs

    Args:
        config_name (str): configuration for an e2e experiment
    """
    # ----------- 1. loading system configs -------------
    # TODO add logs of load changes and config changes
    config_path = os.path.join(
        FINAL_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    pipeline_name = config['pipeline_name']
    pipeline_folder_name = config['pipeline_folder_name']
    node_names = [config['node_name'] for config in config['nodes']]
    adaptation_interval = config['adaptation_interval']
    # first node of the pipeline determins the pipeline data_type
    data_type = config['nodes'][0]['data_type']
    series = config['series']
    pipeline_path = os.path.join(
        PIPLINES_PATH,
        pipeline_folder_name,
        'seldon-core-version'
    )

    dir_path = os.path.join(
        FINAL_RESULTS_PATH,
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

    # ----------- 2. loading profiling configs -------------
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    with open(ACCURACIES_PATH, 'r') as cf:
        accuracies = yaml.safe_load(cf)

    # timeout variable
    timeout = config['timeout']

    # profiling config
    series = config['series']
    number_tasks = config['number_tasks']
    profiling_series = config['profiling_series']
    model_name = config['model_name']
    task_name = config['task_name']
    initial_active_model = config['initial_active_model']
    initial_cpu_allocation = config['initial_cpu_allocation']
    initial_replica = config['initial_replica']
    initial_batch = config['initial_batch']
    scaling_cap = config['scaling_cap']
    pipeline_name = config['pipeline_name']
    only_measured_profiles = config['only_measured_profiles']
    profiling_load = config['profiling_load']

    # pipeline config
    num_state_limit = config['num_state_limit']
    optimization_method = config['optimization_method']
    allocation_mode = config['allocation_mode']
    threshold = config['threshold']
    sla_factor = config['sla_factor']
    accuracy_method = config['accuracy_method']
    normalize_accuracy = config['normalize_accuracy']

    # pipeline accuracy
    pipeline_accuracies = accuracies[pipeline_name]

    # optimizer
    alpha = config['alpha']
    beta = config['beta']
    gamma = config['gamma']

    pipeline = generate_simulated_pipeline(
        number_tasks=number_tasks,
        profiling_series=profiling_series,
        model_names=model_name,
        task_names=task_name,
        initial_active_model=initial_active_model,
        allocation_mode=allocation_mode,
        initial_cpu_allocation=initial_cpu_allocation,
        initial_replica=initial_replica,
        initial_batch=initial_batch,
        threshold=threshold,
        sla_factor=sla_factor,
        accuracy_method=accuracy_method,
        normalize_accuracy=normalize_accuracy,
        pipeline_accuracies=pipeline_accuracies,
        only_measured_profiles=only_measured_profiles,
        profiling_load=profiling_load)

    # should be inside of experiments
    adapter = Adapter(
        pipeline_name=pipeline_name,
        pipeline=pipeline,
        adaptation_interval=adaptation_interval,
        optimization_method=optimization_method,
        allocation_mode=allocation_mode,
        only_measured_profiles=only_measured_profiles,
        scaling_cap=scaling_cap,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_state_limit=num_state_limit
    )

    # ----------- 3. Running an experiment series -------------
    # 1. Setup the pipeline
    # 2. Makes two processes for experiment and adapter
    # 3. Run both processes at the same time
    # 4. Join both processes

    # 0. setup pipeline
    remove_pipeline(pipeline_name=pipeline_name)
    experiments_exist, experiment_id =\
        setup_pipeline(
        pipeline_name=pipeline_name,
        node_names=node_names,
        config=config,
        pipeline_path=pipeline_path,
        data_type=data_type
        )
    
    # TODO check if experiment exists

    # 1. process one the experiment runner
    experiment_process = multiprocessing.Process(
        target=experiments,
        kwargs={
            'pipeline_name': pipeline_name,
            'node_names': node_names,
            'config': config,
            'pipeline_path': pipeline_path,
            'data_type': data_type,
            'experiment_id': experiment_id
        })

    # 2. process two the pipeline adapter
    adaptation_process = multiprocessing.Process(
        target=adapter.start_experiment)

    # start both processese at the same time
    experiment_process.start()
    adaptation_process.start()

    # finish the experiments
    experiment_process.join()
    adaptation_process.join()

    print(f'waiting for timeout: {timeout} seconds')
    for _ in tqdm(range(20)):
        time.sleep((timeout)/20)
    remove_pipeline(pipeline_name=pipeline_name)

if __name__ == "__main__":
    main()
