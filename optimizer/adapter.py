from typing import Dict, Literal, Tuple, Union, Optional
import time
import tqdm
import numpy as np
from kubernetes import config
from kubernetes import client
from typing import List
import os
import sys
import pandas as pd
import concurrent.futures
import tensorflow as tf
from tensorflow.keras.models import load_model


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))
from experiments.utils.pipeline_operations import check_node_up

from experiments.utils.prometheus import PromClient

prom_client = PromClient()

from kubernetes import config
from kubernetes import client

try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)

kube_custom_api = client.CustomObjectsApi()

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from optimizer import Optimizer, Pipeline
from experiments.utils.constants import NAMESPACE, LSTM_PATH, LSTM_INPUT_SIZE
from experiments.utils import logger
from optimizer.optimizer import Optimizer


class Adapter:
    def __init__(
        self,
        pipeline_name: str,
        pipeline: Pipeline,
        node_names: List[str],
        adaptation_interval: int,
        optimization_method: Literal["gurobi", "brute-force"],
        allocation_mode: Literal["base", "variable"],
        only_measured_profiles: bool,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        num_state_limit: int,
        monitoring_duration: int,
        predictor_type: str,
        baseline_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            pipeline_name (str): name of the pipeline
            pipeline (Pipeline): pipeline object
            adaptation_interval (int): adaptation interval of the pipeline
            optimization_method (Literal[gurobi, brute-force])
            allocation_mode (Literal[base;variable])
            only_measured_profiles (bool)
            scaling_cap (int)
            alpha (float): accuracy weight
            beta (float): resource weight
            gamma (float): batching weight
            num_state_limit (int): cap on the number of optimal states
            monitoring_duration (int): the monitoring
                deamon observing duration
        """
        self.pipeline_name = pipeline_name
        self.pipeline = pipeline
        self.node_names = node_names
        self.adaptation_interval = adaptation_interval
        self.optimizer = Optimizer(
            pipeline=pipeline,
            allocation_mode=allocation_mode,
            complete_profile=False,
            only_measured_profiles=only_measured_profiles,
            random_sample=False,
            baseline_mode=baseline_mode,
        )
        self.optimization_method = optimization_method
        self.scaling_cap = scaling_cap
        self.batching_cap = batching_cap
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_state_limit = num_state_limit
        self.monitoring_duration = monitoring_duration
        self.predictor_type = predictor_type
        self.monitoring = Monitoring(pipeline_name=self.pipeline_name)
        self.predictor = Predictor(predictor_type=self.predictor_type)

    def start_adaptation(self):
        # 0. Check if pipeline is up
        # 1. Use monitoring for periodically checking the status of
        #     the pipeline in terms of load
        # 2. Watches the incoming load in the system
        # 3. LSTM for predicting the load
        # 4. Get the existing pipeline state, batch size, model variant and replicas per
        #     each node
        # 5. Give the load and pipeline status to the optimizer
        # 6. Compare the optimal solutions from the optimzer
        #     to the existing pipeline's state
        # 7. Use the change config script to change the pipelien to the new config

        time_interval = 0
        timestep = 0
        pipeline_up = False
        pipeline_up = check_node_up(node_name="router")
        # TODO add the check of whether enough time has
        # passed to start adaptation or not
        if pipeline_up:
            initial_config = self.extract_config()
            self.monitoring.adaptation_step_report(
                to_apply_config=initial_config,
                objective=None,
                timestep=timestep,
                time_interval=time_interval,
                predicted_load=0,
            )
        while True:
            logger.info(f"Waiting {self.adaptation_interval}" " to make next descision")
            for _ in tqdm.tqdm(range(self.adaptation_interval)):
                time.sleep(1)
            pipeline_up = check_node_up(node_name="router")
            if not pipeline_up:
                logger.info(
                    "no pipeline in the system," " aborting adaptation process ..."
                )
                # with the message that the process has ended
                break
            time_interval += self.adaptation_interval
            timestep += 1
            rps_series = self.monitoring.rps_monitor(
                monitoring_duration=self.monitoring_duration
            )
            predicted_load = round(self.predictor.predict(rps_series))
            logger.info(f"\nPredicted Load: {predicted_load}\n")
            optimal = self.optimizer.optimize(
                optimization_method=self.optimization_method,
                scaling_cap=self.scaling_cap,
                batching_cap=self.batching_cap,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                arrival_rate=predicted_load,
                num_state_limit=self.num_state_limit,
            )
            objective_value = optimal["objective"][0]
            new_configs = self.output_parser(optimal)
            to_apply_config = self.choose_config(new_configs)
            self.change_pipeline_config(to_apply_config)
            self.monitoring.adaptation_step_report(
                to_apply_config=to_apply_config,
                objective=objective_value,
                timestep=timestep,
                time_interval=time_interval,
                predicted_load=predicted_load,
            )

    def output_parser(self, optimizer_output: pd.DataFrame):
        new_configs = []
        for _, row in optimizer_output.iterrows():
            config = {}
            for task_id, task_name in enumerate(self.node_names):
                config[task_name] = {}
                config[task_name]["cpu"] = row[f"task_{task_id}_cpu"]
                config[task_name]["replicas"] = int(row[f"task_{task_id}_replicas"])
                config[task_name]["batch"] = int(row[f"task_{task_id}_batch"])
                config[task_name]["variant"] = row[f"task_{task_id}_variant"]
            new_configs.append(config)
        return new_configs

    def choose_config(self, new_configs: List[Dict[str, Dict[str, Union[str, int]]]]):
        # This should be from comparing with the
        # current config
        # easiest for now is to choose config with
        # with the least change from former config
        current_config = self.extract_config()
        new_config_socres = []
        for new_config in new_configs:
            new_config_score = 0
            for node_name, new_node_config in new_config.items():
                for config_knob, config_value in new_node_config.items():
                    if (
                        config_knob == "variant"
                        and config_value != current_config[node_name][config_knob]
                    ):
                        new_config_score -= 1
                    if (
                        config_knob == "batch"
                        and str(config_value) != current_config[node_name][config_knob]
                    ):
                        new_config_score -= 1
            new_config_socres.append(new_config_score)
        chosen_config_index = new_config_socres.index(max(new_config_socres))
        chosen_config = new_configs[chosen_config_index]
        return chosen_config

    def extract_config(self):
        current_config = {}
        for node_name in self.node_names:
            node_config = {}
            raw_config = kube_custom_api.get_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=NAMESPACE,
                plural="seldondeployments",
                name=node_name,
            )
            component_config = raw_config["spec"]["predictors"][0]["componentSpecs"][0]
            env_vars = component_config["spec"]["containers"][0]["env"]
            replicas = component_config["replicas"]
            cpu = int(
                component_config["spec"]["containers"][0]["resources"]["requests"][
                    "cpu"
                ]
            )
            for env_var in env_vars:
                if env_var["name"] == "MODEL_VARIANT":
                    variant = env_var["value"]
                if env_var["name"] == "MLSERVER_MODEL_MAX_BATCH_SIZE":
                    batch = env_var["value"]
            node_config["replicas"] = replicas
            node_config["variant"] = variant
            node_config["batch"] = batch
            node_config["cpu"] = cpu
            current_config[node_name] = node_config
        return current_config

    def change_pipeline_config(self, config: Dict[str, Dict[str, int]]):
        """change the existing configuration based on the optimizer
           output
        Args:
            config (Dict[str, Dict[str, int]]): _description_
        """
        node_names = list(config.keys())
        node_configs = list(config.values())
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self.change_node_config, zip(node_names, node_configs))
            )
        return results

    def change_node_config(self, inputs: Tuple[str, Dict[str, int]]):
        node_name, node_config = inputs
        deployment_config = kube_custom_api.get_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=NAMESPACE,
            plural="seldondeployments",
            name=node_name,
        )
        deployment_config["spec"]["predictors"][0]["componentSpecs"][0][
            "replicas"
        ] = node_config["replicas"]
        deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
            "containers"
        ][0]["resources"]["limits"]["cpu"] = str(node_config["cpu"])
        deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
            "containers"
        ][0]["resources"]["requests"]["cpu"] = str(node_config["cpu"])
        for env_index, env_var in enumerate(
            deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["env"]
        ):
            if env_var["name"] == "MODEL_VARIANT":
                deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                    "containers"
                ][0]["env"][env_index]["value"] = node_config["variant"]
            if env_var["name"] == "MLSERVER_MODEL_MAX_BATCH_SIZE":
                deployment_config["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                    "containers"
                ][0]["env"][env_index]["value"] = str(node_config["batch"])
        kube_custom_api.replace_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=NAMESPACE,
            plural="seldondeployments",
            name=node_name,
            body=deployment_config,
        )
        return True


class Monitoring:
    def __init__(self, pipeline_name) -> None:
        self.pipeline_name = pipeline_name
        self.adaptation_report = {}

    def rps_monitor(self, monitoring_duration: int = 1) -> List[int]:
        """
        Get the rps of the router
        duration in minutes
        """
        rate = 15
        rps_series, _ = prom_client.get_request_per_second(
            pod_name="router",
            namespace="default",
            duration=monitoring_duration,
            container="router",
            rate=rate,
        )
        return rps_series

    def adaptation_step_report(
        self,
        to_apply_config: Dict[str, Dict[str, Union[str, int]]],
        objective: float,
        timestep: int,
        time_interval: int,
        predicted_load: int,
    ):
        self.adaptation_report[timestep] = {}
        self.adaptation_report[timestep]["config"] = to_apply_config
        self.adaptation_report[timestep]["objective"] = objective
        self.adaptation_report[timestep]["time_interval"] = time_interval
        self.adaptation_report[timestep]["predicted_load"] = predicted_load


class Predictor:
    def __init__(
            self, predictor_type, backup_predictor:str = 'reactive') -> int:
        self.predictor_type = predictor_type
        self.backup_predictor = backup_predictor
        predictors = {
            'lstm': load_model(LSTM_PATH),
            'reactive': lambda l: l[-1],
            'max': lambda l: max(l),
            'avg': lambda l: max(l) / len(l)
        }
        self.model = predictors[predictor_type]
        self.backup_model = predictors[backup_predictor]

    def predict(self, series: List[int]):
        series_minutes = []
        step = 60
        for i in range(0, len(series), step):
            series_minutes.append(max(series[i : i + step]))
        if self.predictor_type == "lstm":
            if len(series_minutes) < LSTM_INPUT_SIZE:
                logger.info(
                    'not enough information for lstm'
                    f' usting backup predictor {self.backup_predictor}')
                return self.backup_model(series_minutes)
            model_intput = tf.convert_to_tensor(
                np.array(series_minutes).reshape((-1, LSTM_INPUT_SIZE, 1)),
                dtype=tf.float32,
            )
            model_output = self.model.predict(model_intput)[0][0]
        else:
            model_output = self.model(series)

        return model_output
