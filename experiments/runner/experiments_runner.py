"""
Iterate through all possible combination
of pipelines
"""
import os
import json
import yaml
import click
import sys
import shutil
from barazmoon.twitter import twitter_workload_generator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from experiments.utils.prometheus import PromClient

prom_client = PromClient()

from experiments.utils.pipeline_operations import (
    load_data,
    warm_up,
    check_load_test,
    load_test,
    remove_pipeline,
    setup_router_pipeline
)

from experiments.utils.constants import (
    PIPLINES_PATH,
    FINAL_CONFIGS_PATH,
    FINAL_RESULTS_PATH,
)
from experiments.utils import logger


def setup_pipeline(
    pipeline_name: str,
    node_names: str,
    config: dict,
    pipeline_path: str,
    data_type: str,
):
    timeout = config["timeout"]

    model_variants = []
    max_batch_sizes = []
    max_batch_times = []
    cpu_requests = []
    memory_requests = []
    replicas = []
    use_threading = []
    num_iterop_threads = []
    num_threads = []
    for node_config in config["nodes"]:
        model_variants.append(node_config["model_variants"])
        max_batch_sizes.append(node_config["max_batch_size"])
        max_batch_times.append(node_config["max_batch_time"])
        cpu_requests.append(node_config["cpu_request"])
        memory_requests.append(node_config["memory_request"])
        replicas.append(node_config["replicas"])
        use_threading.append([node_config["use_threading"]][0])
        num_iterop_threads.append(node_config["num_interop_threads"])
        num_threads.append(node_config["num_threads"])

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
        num_nodes=len(config["nodes"]),
        use_threading=use_threading,
        # HACK for now we set the number of requests
        # proportional to the the number threads
        num_interop_threads=cpu_requests,
        num_threads=cpu_requests,
    )
    logger.info(f"Checking if the model is up ...")
    logger.info("\n")
    # check if the model is up or not
    check_load_test(
        pipeline_name="router",
        model="router",
        data_type=data_type,
        pipeline_path=pipeline_path,
    )
    # logger.info("model warm up ...")
    # logger.info("\n")
    # warm_up_duration = 10
    # warm_up(
    #     pipeline_name="router",
    #     model="router",
    #     data_type=data_type,
    #     pipeline_path=pipeline_path,
    #     warm_up_duration=warm_up_duration,
    # )


def experiments(config: dict, pipeline_path: str, data_type: str):
    logger.info("\n")
    logger.info("-" * 25 + f"starting load test " + "-" * 25)
    logger.info("\n")

    mode = config["mode"]
    benchmark_duration = config["benchmark_duration"]
    workload_type = config["workload_type"]
    workload_config = config["workload_config"]

    if workload_type == "static":
        loads_to_test = workload_config["loads_to_test"]
        load_duration = workload_config["load_duration"]
        workload = [loads_to_test] * load_duration
    elif workload_type == "twitter":
        loads_to_test = []
        for w_config in workload_config:
            damping_factor = w_config["damping_factor"]
            start = w_config["start"]
            end = w_config["end"]
            load_to_test = start + "-" + end
            loads_to_test.append(load_to_test)
        workload = twitter_workload_generator(
            loads_to_test[0], damping_factor=damping_factor
        )
        load_duration = len(workload)

    data = load_data(data_type, pipeline_path)
    # try:
    start_time_experiment, end_time_experiment, responses = load_test(
        pipeline_name="router",
        model="router",
        data_type=data_type,
        data=data,
        workload=workload,
        mode=mode,
        namespace="default",
        benchmark_duration=benchmark_duration,
    )
    logger.info("-" * 25 + "saving the report" + "-" * 25)
    logger.info("\n")
    results = {
        "responses": responses,
        "start_time_experiment": start_time_experiment,
        "end_time_experiment": end_time_experiment,
    }
    return results


@click.command()
@click.option("--config-name", required=True, type=str, default="video")
@click.option(
    "--type-of",
    required=True,
    type=click.Choice(["experiment", "adaptation"]),
    default="experiment",
)
def main(config_name: str, type_of: str):
    """loading system configs

    Args:
        config_name (str): configuration for an e2e experiment
    """
    # ----------- 1. loading system configs -------------
    config_path = os.path.join(FINAL_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)
    series = config["series"]
    # name resuls zero for consistency with the profiling parser
    save_path = os.path.join(FINAL_RESULTS_PATH, "series", str(series), "0.json")
    pipeline_name = config["pipeline_name"]
    pipeline_folder_name = config["pipeline_folder_name"]
    node_names = [config["node_name"] for config in config["nodes"]]
    # first node of the pipeline determins the pipeline data_type
    data_type = config["nodes"][0]["data_type"]
    pipeline_path = os.path.join(
        PIPLINES_PATH, pipeline_folder_name, "seldon-core-version"
    )

    dir_path = os.path.join(FINAL_RESULTS_PATH, "series", str(series))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        dest_config_path = os.path.join(dir_path, "0.yaml")
        shutil.copy(config_path, dest_config_path)
    else:
        num_configs = 0
        # Iterate directory
        for file in os.listdir(dir_path):
            # check only text files
            if file.endswith(".yaml"):
                num_configs += 1
        dest_config_path = os.path.join(dir_path, f"{num_configs}.yaml")
        shutil.copy(config_path, dest_config_path)

    # ----------- 2. loading profiling configs -------------
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)

    # profiling config
    series = config["series"]
    pipeline_name = config["pipeline_name"]

    # ----------- 3. Running an experiment series -------------
    # 1. Setup the pipeline
    # 2. Makes two processes for experiment and adapter
    # 3. Run both processes at the same time
    # 4. Join both processes

    # 0. setup pipeline
    if type_of == "experiment":
        remove_pipeline(pipeline_name=pipeline_name)
        setup_pipeline(
            pipeline_name=pipeline_name,
            node_names=node_names,
            config=config,
            pipeline_path=pipeline_path,
            data_type=data_type,
        )

        # 1. process one the experiment runner
        result = experiments(
            config=config, pipeline_path=pipeline_path, data_type=data_type
        )

        with open(save_path, "w") as outfile:
            outfile.write(json.dumps(result))

    remove_pipeline(pipeline_name=pipeline_name)


if __name__ == "__main__":
    main()
