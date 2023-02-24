import pandas as pd
import click
import time
import os
import sys
import shutil
from typing import List, Dict
import yaml
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from models import (
    Model,
    ResourceAllocation,
    Profile,
    Task,
    Pipeline
)
from optimizer import Optimizer

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..')))

from experiments.utils.constants import (
    PIPELINE_SIMULATION_CONFIGS_PATH,
    PIPELINE_SIMULATION_RESULTS_PATH,
    NODE_PROFILING_RESULTS_STATIC_PATH,
    ACCURACIES_PATH
)
from experiments.utils.loader import Loader


config_key_mapper = "key_config_mapper.csv"

# TEMP all model profiles built using load of 1
def load_profile(series, model_name, experiment_id=1, load=1):
    series_path = os.path.join(
        NODE_PROFILING_RESULTS_STATIC_PATH,
        'series',
        str(series))
    loader = Loader(
        series_path=series_path, config_key_mapper=config_key_mapper,
        model_name=model_name)
    key_config_df = loader.key_config_mapper()
    experiment_ids = key_config_df[
        (key_config_df['load'] == load)]['experiment_id'].tolist()
    metadata_columns = [
        'model_variant',
        'cpu_request',
        'max_batch_size',
        'load']
    results_columns = [
        'model_latencies_min',
        'model_latencies_p99',
        'cpu_usage_count_avg',
        'model_latencies_avg']
    profiling_info = loader.table_maker(
        experiment_ids=experiment_ids,
        metadata_columns=metadata_columns,
        results_columns=results_columns)
    return profiling_info

def read_task_profiles(
    profiling_info: pd.DataFrame,
    task_accuracies: Dict[str, float]) -> List[Model]:
    available_model_profiles = []
    for model_variant in profiling_info['model_variant'].unique():
        model_variant_profiling_info =\
            profiling_info[profiling_info['model_variant'] == model_variant]
        model_variant_profiling_info.sort_values(by=['max_batch_size', 'cpu_request'])
        for cpu_request in model_variant_profiling_info['cpu_request'].unique():
            cpu_request_profiling_info =\
                model_variant_profiling_info[
                    model_variant_profiling_info['cpu_request'] == cpu_request]
            measured_profiles = []
            for _, row in cpu_request_profiling_info.iterrows():
                # TODO throughput from profiling
                measured_profiles.append(
                    Profile(
                        batch=row['max_batch_size'],
                        latency=row['model_latencies_avg']))
            available_model_profiles.append(
                Model(
                    name=model_variant,
                    resource_allocation=ResourceAllocation(
                        cpu=cpu_request),
                    measured_profiles=measured_profiles,
                    accuracy=task_accuracies[model_variant]
                ) 
            )
    return available_model_profiles

def generate_pipeline(
    number_tasks: int,
    profiling_series: List[int],
    model_name: List[str],
    task_names: List[str],
    experiment_id: List[int],
    initial_active_model: List[str],
    initial_cpu_allocation: List[int],
    initial_replica: List[int],
    initial_batch: List[int],
    allocation_mode: str,
    threshold: int,
    sla_factor: int,
    accuracy_method: int,
    normalize_accuracy: bool,
    pipeline_accuracies: Dict[str, Dict[str, float]]
    ) -> Pipeline:
    inference_graph = []
    for i in range(number_tasks):
        profiling_info = load_profile(
            series=profiling_series[i], model_name=model_name[i],
            experiment_id=experiment_id[i])
        available_model_profiles =\
            read_task_profiles(
                profiling_info=profiling_info,
                task_accuracies=pipeline_accuracies[task_names[i]])
        task = Task(
            name=task_names[i],
            available_model_profiles = available_model_profiles,
            active_variant = initial_active_model[i],
            active_allocation=ResourceAllocation( 
                cpu=initial_cpu_allocation[i]),
            replica=initial_replica[i],
            batch=initial_batch[i],
            threshold=threshold,
            sla_factor=sla_factor,
            allocation_mode=allocation_mode,
            normalize_accuracy=normalize_accuracy,
            gpu_mode=False,
        )
        inference_graph.append(task)
    pipeline = Pipeline(
        inference_graph=inference_graph,
        gpu_mode=False,
        sla_factor=sla_factor,
        accuracy_method=accuracy_method,
        normalize_accuracy=normalize_accuracy
    )
    return pipeline

@click.command()
@click.option(
    '--config-name', required=True, type=str, default='video-pipeline')
def main(config_name: str):
    config_path = os.path.join(
        PIPELINE_SIMULATION_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    with open(ACCURACIES_PATH, 'r') as cf:
        accuracies = yaml.safe_load(cf)
    # profiling config
    series = config['series']
    number_tasks = config['number_tasks']
    profiling_series = config['profiling_series']
    model_name = config['model_name']
    task_name = config['task_name']
    experiment_id = config['experiment_id']
    initial_active_model = config['initial_active_model']
    initial_cpu_allocation = config['initial_cpu_allocation']
    initial_replica = config['initial_replica']
    initial_batch = config['initial_batch']
    scaling_cap = config['scaling_cap']
    pipeline_name = config['pipeline_name']
    complete_profile = config['complete_profile']

    # pipeline config
    arrival_rate = config['arrival_rate']
    num_state_limit = config['num_state_limit']
    generate = config['generate']
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

    # config generation config
    dir_path = os.path.join(
        PIPELINE_SIMULATION_RESULTS_PATH,
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

    pipeline = generate_pipeline(
        number_tasks=number_tasks,
        profiling_series=profiling_series,
        model_name=model_name,
        task_names=task_name,
        experiment_id=experiment_id,
        initial_active_model=initial_active_model,
        allocation_mode=allocation_mode,
        initial_cpu_allocation=initial_cpu_allocation,
        initial_replica=initial_replica,
        initial_batch=initial_batch,
        threshold=threshold,
        sla_factor=sla_factor,
        accuracy_method=accuracy_method,
        normalize_accuracy=normalize_accuracy,
        pipeline_accuracies=pipeline_accuracies)

    optimizer = Optimizer(
        pipeline=pipeline,
        allocation_mode=allocation_mode,
        complete_profile=complete_profile
    )

    all_states_time = None
    feasible_time = None
    optimal_time = None
    total_time = None
    time_file = open(
        os.path.join(dir_path, 'times.csv'), "w")
    if optimization_method == 'gurobi':
        assert generate[0] == 'optimal', 'only optimal is allowed with gurbi'
    total_time = time.time()
    if 'all' in generate:
        all_states_time = time.time()
        # all states
        states = optimizer.all_states(
            check_constraints=False,
            scaling_cap=scaling_cap,
            alpha=alpha, beta=beta, gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit)
        states.to_markdown(
            os.path.join(
                dir_path, 'readable-all-states.csv'), index=False)
        states.to_csv(
            os.path.join(dir_path, 'all-states.csv'), index=False)
        all_states_time = time.time() - all_states_time
        time_file.write(f'all: {all_states_time}\n')
        print(f"all states time: {all_states_time}")
    if 'feasible' in generate:
        feasible_time = time.time()
        # all feasibla states
        with_constraints = optimizer.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha, beta=beta, gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit)
        # print(f"{with_constraints = }")
        with_constraints.to_markdown(
            os.path.join(
                dir_path, 'readable-with-constraints.csv'),
                index=False)
        with_constraints.to_csv(
            os.path.join(dir_path, 'with-constraints.csv'), index=False)
        feasible_time = time.time() - feasible_time
        time_file.write(f'feasible_time: {feasible_time}\n')
        print(f"with constraint time: {feasible_time}")
    if 'optimal' in generate:
        if optimization_method == 'gurobi':
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method=optimization_method,
                scaling_cap=scaling_cap,
                alpha=alpha, beta=beta, gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
            # print(f"{optimal = }")
            optimal.to_markdown(os.path.join(
                dir_path, 'readable-optimal-gurobi.csv'), index=False)
            optimal.to_csv(os.path.join(
                dir_path, 'optimal-gurobi.csv'), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f'optimal_time_gurobi: {optimal_time}\n')
            print(f"optimal time gurobi: {optimal_time}")

        if optimization_method == 'brute-force':
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method='brute-force',
                scaling_cap=scaling_cap,
                alpha=alpha, beta=beta, gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
            # print(f"{optimal = }")
            optimal.to_markdown(os.path.join(
                dir_path, 'readable-optimal-brute-force.csv'), index=False)
            optimal.to_csv(os.path.join(
                dir_path, 'optimal-brute-force.csv'), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f'optimal_time_brute_force: {optimal_time}\n')
            print(f"optimal time brute-force: {optimal_time}")

        if optimization_method == 'both':
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method='brute-force',
                scaling_cap=scaling_cap,
                alpha=alpha, beta=beta, gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
            # print(f"{optimal = }")
            optimal.to_markdown(os.path.join(
                dir_path, 'readable-optimal-brute-force.csv'), index=False)
            optimal.to_csv(os.path.join(
                dir_path, 'optimal-brute-force.csv'), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f'optimal_time_brute_force: {optimal_time}\n')
            print(f"optimal time brute-force: {optimal_time}")

            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method='gurobi',
                scaling_cap=scaling_cap,
                alpha=alpha, beta=beta, gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
            # print(f"{optimal = }")
            optimal.to_markdown(os.path.join(
                dir_path, 'readable-optimal-gurobi.csv'), index=False)
            optimal.to_csv(os.path.join(
                dir_path, 'optimal-gurobi.csv'), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f'optimal_time_gurobi: {optimal_time}\n')
            print(f"optimal time gurobi: {optimal_time}")

    total_time = time.time() - total_time
    time_file.write(f'total_time: {total_time}')
    time_file.close()
    print(f"total time spent: {total_time}")

if __name__ == "__main__":
    main()
