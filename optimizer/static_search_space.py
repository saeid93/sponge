import pandas as pd
import click
import numpy as np
import os
import sys
import shutil
from typing import List
import yaml
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from simulator import (
    Model,
    ResourceAllocation,
    Profile,
    Task,
    Pipeline,
    Optimizer)

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..')))

from experiments.utils.constants import (
    PIPELINE_SIMULATION_CONFIGS_PATH,
    PIPELINE_SIMULATION_RESULTS_PATH,
    NODE_PROFILING_RESULTS_STATIC_PATH
)
from experiments.utils.loader import Loader


config_key_mapper = "key_config_mapper.csv"

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

# TODO add accuracies
# TODO polish file format
def read_task_profiles(profiling_info: pd.DataFrame) -> List[Model]:
    available_model_profiles = []
    for model_variant in profiling_info['model_variant'].unique():
        model_variant_profiling_info =\
            profiling_info[profiling_info['model_variant'] == model_variant]
        for cpu_request in model_variant_profiling_info['cpu_request'].unique():
            cpu_request_profiling_info =\
                model_variant_profiling_info[
                    model_variant_profiling_info['cpu_request'] == cpu_request]
            measured_profiles = []
            for _, row in cpu_request_profiling_info.iterrows():
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
                    accuracy=0.5
                ) 
            )
    return available_model_profiles

def generate_pipeline(
    number_tasks: int,
    profiling_series: List[int],
    model_name: List[str],
    task_name: List[str],
    experiment_id: List[int],
    initial_active_model: List[str],
    initial_cpu_allocation: List[int],
    initial_replica: List[int],
    initial_batch: List[int]
    ) -> Pipeline:
    inference_graph = []
    for i in range(number_tasks):
        profiling_info = load_profile(
            series=profiling_series[i], model_name=model_name[i],
            experiment_id=experiment_id[i])
        available_model_profiles =\
            read_task_profiles(profiling_info=profiling_info)
        task = Task(
            name=task_name[i],
            available_model_profiles = available_model_profiles,
            active_variant = initial_active_model[i],
            active_allocation=ResourceAllocation(
                cpu=initial_cpu_allocation[i]),
            replica=initial_replica[i],
            batch=initial_batch[i],
            gpu_mode=False
        )
        inference_graph.append(task)
    pipeline = Pipeline(
        inference_graph=inference_graph,
        gpu_mode=False
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

    # pipeline config
    sla = config['sla']
    arrival_rate = config['arrival_rate']
    limit = config['limit']
    generate = config['generate']

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
        task_name=task_name,
        experiment_id=experiment_id,
        initial_active_model=initial_active_model,
        initial_cpu_allocation=initial_cpu_allocation,
        initial_replica=initial_replica,
        initial_batch=initial_batch)

    optimizer = Optimizer(
        pipeline=pipeline
    )

    if 'all' in generate:
        # all states
        states = optimizer.all_states(scaling_cap=scaling_cap, limit=limit)
        print(f"{states = }")
        states.to_markdown(os.path.join(dir_path, 'all-states.md'))
        states.to_csv(os.path.join(dir_path, 'all-states.csv'))
    if 'feasible' in generate:
        # all feasibla states
        with_constraints = optimizer.all_states(
            check_constraints=True, scaling_cap=scaling_cap,
            arrival_rate=arrival_rate, sla=sla, limit=limit)
        print(f"{with_constraints = }")
        with_constraints.to_markdown(
            os.path.join(dir_path, 'with-constraints.md'))
        with_constraints.to_csv(
            os.path.join(dir_path, 'with-constraints.csv'))
    if 'optimal' in generate:
        # optimal states
        optimal = optimizer.greedy_optimizer(
            scaling_cap=scaling_cap, sla=sla, arrival_rate=arrival_rate, limit=limit)
        print(f"{optimal = }")
        optimal.to_markdown(os.path.join(dir_path, 'optimal.md'))
        optimal.to_csv(os.path.join(dir_path, 'optimal.csv'))

if __name__ == "__main__":
    main()
