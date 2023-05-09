import pandas as pd
import os
import sys
from typing import List, Dict, Literal
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from optimizer import Model, ResourceAllocation, Profile, Task, Pipeline

from experiments.utils.constants import (
    NODE_PROFILING_RESULTS_PATH,
)
from experiments.utils.parser import Parser

config_key_mapper = "key_config_mapper.csv"


def load_profile(series: int, model_name: str, load: int = 1):
    """load a node profile

    Args:
        series (int): series of profiled data
        model_name (str): name of the model
        load (int): The load for profiled data. Defaults to 1.

    Returns:
        _type_: _description_
    """
    series_path = os.path.join(NODE_PROFILING_RESULTS_PATH, "series", str(series))
    loader = Parser(
        series_path=series_path,
        config_key_mapper=config_key_mapper,
        model_name=model_name,
    )
    key_config_df = loader.key_config_mapper()
    experiment_ids = key_config_df[
        (key_config_df["load"] == load) | (key_config_df["load"] == str(load))
    ]["experiment_id"].tolist()
    metadata_columns = ["model_variant", "cpu_request", "max_batch_size", "load"]
    results_columns = [
        "throughput_max",
        "model_latencies_min",
        "model_latencies_p99",
        "cpu_usage_count_avg",
        "model_latencies_avg",
    ]
    profiling_info = loader.table_maker(
        experiment_ids=experiment_ids,
        metadata_columns=metadata_columns,
        results_columns=results_columns,
    )
    return profiling_info


def make_task_profiles(
    profiling_info: pd.DataFrame,
    task_accuracies: Dict[str, float],
    only_measured_profiles: bool,
) -> List[Model]:
    """make a task all model variants profiles

    Args:
        profiling_info (pd.DataFrame): dataframe of profiles
        task_accuracies (Dict[str, float]): accuracies of each
            variant of the task
        only_measured_profiles (bool): whther to use regression
            or not

    Returns:
        List[Model]: list of all the model variants for a task
    """
    available_model_profiles = []
    for model_variant in profiling_info["model_variant"].unique():
        model_variant_profiling_info = profiling_info[
            profiling_info["model_variant"] == model_variant
        ]
        model_variant_profiling_info.sort_values(by=["max_batch_size", "cpu_request"])
        for cpu_request in model_variant_profiling_info["cpu_request"].unique():
            cpu_request_profiling_info = model_variant_profiling_info[
                model_variant_profiling_info["cpu_request"] == cpu_request
            ]
            measured_profiles = []
            for _, row in cpu_request_profiling_info.iterrows():
                # throughput from profiling
                if only_measured_profiles:
                    measured_profiles.append(
                        Profile(
                            batch=row["max_batch_size"],
                            latency=row["model_latencies_avg"],
                            measured_throughput=row["throughput_max"],
                        )
                    )
                else:
                    # throughput from fromulation
                    measured_profiles.append(
                        Profile(
                            batch=row["max_batch_size"],
                            latency=row["model_latencies_avg"],
                        )
                    )
            if None in list(map(lambda l: l.latency, measured_profiles)):
                # skipping unresponsive profiles
                a = 1
                continue
            available_model_profiles.append(
                Model(
                    name=model_variant,
                    resource_allocation=ResourceAllocation(cpu=cpu_request),
                    measured_profiles=measured_profiles,
                    accuracy=task_accuracies[model_variant],
                    only_measured_profiles=only_measured_profiles,
                )
            )
    return available_model_profiles


def generate_simulated_pipeline(
    number_tasks: int,
    profiling_series: List[int],
    model_names: List[str],
    task_names: List[str],
    initial_active_model: List[str],
    initial_cpu_allocation: List[int],
    initial_replica: List[int],
    initial_batch: List[int],
    allocation_mode: Literal["base", "variable"],
    threshold: int,
    sla_factor: int,
    accuracy_method: Literal["sum", "average", "multiply"],
    normalize_accuracy: bool,
    pipeline_accuracies: Dict[str, Dict[str, float]],
    only_measured_profiles: bool,
    profiling_load: bool,
) -> Pipeline:
    """generates simulated version of the pipelines
       profiles

    Args:
        number_tasks (int): number of tasks in the
            linear pipeline
        profiling_series (List[int]): profiling series
        model_names (List[str]): name of the model tasks
            (can be similar to task_names)
        task_names (List[str]): name of the model tasks
            (can be similar to model_names)
        initial_active_model (List[str]): initially active model for each step
        initial_cpu_allocation (List[int]): initial allocation for each step
        initial_replica (List[int]): initial number of replicas for each stage
        initial_batch (List[int]): initial batch size for each stage
        allocation_mode (Literal['base', 'variable']): allocation type
        threshold (int): threshold for choosing base allocations
        sla_factor (int): computing sla based on the slowest moel
        accuracy_method (Literal['sum', 'average', 'multiply']): accuracy calculation method
        normalize_accuracy (bool): actual accuracy values or normalizing them
        pipeline_accuracies (Dict[str, Dict[str, float]]): all the accuracies of each node
            of pipeline
        only_measured_profiles (bool): whether to use regression or not
        profiling_load (bool): the load in the profling used for making profiles

    Returns:
        Pipeline: _description_
    """
    inference_graph = []
    for i in range(number_tasks):
        profiling_info = load_profile(
            series=profiling_series[i], model_name=model_names[i], load=profiling_load
        )
        available_model_profiles = make_task_profiles(
            profiling_info=profiling_info,
            task_accuracies=pipeline_accuracies[task_names[i]],
            only_measured_profiles=only_measured_profiles,
        )
        task = Task(
            name=task_names[i],
            available_model_profiles=available_model_profiles,
            active_variant=initial_active_model[i],
            active_allocation=ResourceAllocation(cpu=initial_cpu_allocation[i]),
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
        normalize_accuracy=normalize_accuracy,
    )
    return pipeline
