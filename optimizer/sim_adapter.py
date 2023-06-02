from typing import Dict, Literal, Union, Optional, Any
import numpy as np
from typing import List
import os
import sys
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras.models import load_model


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from optimizer import Optimizer, Pipeline
from experiments.utils.constants import LSTM_PATH, LSTM_INPUT_SIZE
from experiments.utils import logger
from optimizer.optimizer import Optimizer


class SimAdapter:
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
        backup_predictor_type: str = "max",
        backup_predictor_duration: int = 2,
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
        self.backup_predictor_type = backup_predictor_type
        self.backup_predictor_duration = backup_predictor_duration
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
        self.monitoring = Monitoring(
            pipeline_name=self.pipeline_name, sla=self.pipeline.sla
        )
        self.predictor = Predictor(
            predictor_type=self.predictor_type,
            backup_predictor_type=self.backup_predictor_type,
            backup_predictor_duration=self.backup_predictor_duration,
        )

    def start_adaptation(
        self, workload: List[int], initial_config: Dict[str, Dict[str, Union[str, int]]]
    ):
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
        old_config = deepcopy(initial_config)
        for timestep in range(
            self.adaptation_interval, len(workload), self.adaptation_interval
        ):
            time_interval += self.adaptation_interval
            to_apply_config = None
            to_save_config = None
            objective = None

            rps_series = workload[
                max(0, timestep - self.monitoring_duration * 60) : timestep
            ]
            self.update_recieved_load(all_recieved_loads=rps_series)
            predicted_load = round(self.predictor.predict(rps_series))
            logger.info("-" * 50)
            logger.info(f"\nPredicted Load: {predicted_load}\n")
            logger.info("-" * 50)
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
            # optimal = pd.DataFrame() # TEMP
            if "objective" in optimal.columns:
                objective = optimal["objective"][0]
                new_configs = self.output_parser(optimal)
                to_apply_config = self.choose_config(new_configs, old_config)
            if to_apply_config is not None:
                to_save_config = self.saving_config_builder(
                    to_apply_config=deepcopy(to_apply_config),
                    node_orders=deepcopy(self.node_names),
                    stage_wise_latencies=deepcopy(self.pipeline.stage_wise_latencies),
                    stage_wise_accuracies=deepcopy(self.pipeline.stage_wise_accuracies),
                    stage_wise_throughputs=deepcopy(
                        self.pipeline.stage_wise_throughput
                    ),
                )
            self.monitoring.adaptation_step_report(
                to_apply_config=to_save_config,
                objective=objective,
                timestep=timestep,
                time_interval=time_interval,
                monitored_load=rps_series,
                predicted_load=predicted_load,
            )
            old_config = deepcopy(to_apply_config)

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

    def choose_config(
        self, new_configs: List[Dict[str, Dict[str, Union[str, int]]]], current_config
    ):
        # This should be from comparing with the
        # current config
        # easiest for now is to choose config with
        # with the least change from former config
        if current_config is None:
            # if the current config is None just return the first config
            return new_configs[0]
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

    def update_recieved_load(self, all_recieved_loads) -> None:
        """extract the entire sent load during the
        experiment
        """
        self.monitoring.update_recieved_load(all_recieved_loads)

    def saving_config_builder(
        self,
        to_apply_config: Dict[str, Any],
        node_orders: List[str],
        stage_wise_latencies: List[float],
        stage_wise_accuracies: List[float],
        stage_wise_throughputs: List[float],
    ):
        saving_config = to_apply_config
        for index, node in enumerate(node_orders):
            saving_config[node]["latency"] = stage_wise_latencies[index]
            saving_config[node]["accuracy"] = stage_wise_accuracies[index]
            saving_config[node]["throughput"] = stage_wise_throughputs[index]
        return saving_config


class Monitoring:
    def __init__(self, pipeline_name: str, sla: float) -> None:
        self.pipeline_name = pipeline_name
        self.adaptation_report = {}
        self.adaptation_report["timesteps"] = {}
        self.adaptation_report["metadata"] = {}
        self.adaptation_report["metadata"]["sla"] = sla

    def adaptation_step_report(
        self,
        to_apply_config: Dict[str, Dict[str, Union[str, int]]],
        objective: float,
        timestep: str,
        time_interval: int,
        monitored_load: List[int],
        predicted_load: int,
    ):
        timestep = int(timestep)
        self.adaptation_report["timesteps"][timestep] = {}
        self.adaptation_report["timesteps"][timestep]["config"] = to_apply_config
        self.adaptation_report["timesteps"][timestep]["objective"] = objective
        self.adaptation_report["timesteps"][timestep]["time_interval"] = time_interval
        self.adaptation_report["timesteps"][timestep]["monitored_load"] = monitored_load
        self.adaptation_report["timesteps"][timestep]["predicted_load"] = predicted_load

    def update_recieved_load(self, all_recieved_loads: List[float]):
        self.adaptation_report["metadata"]["recieved_load"] = all_recieved_loads


class Predictor:
    def __init__(
        self,
        predictor_type,
        backup_predictor_type: str = "reactive",
        backup_predictor_duration=2,
    ) -> int:
        self.predictor_type = predictor_type
        self.backup_predictor = backup_predictor_type
        predictors = {
            "lstm": load_model(LSTM_PATH),
            "reactive": lambda l: l[-1],
            "max": lambda l: max(l),
            "avg": lambda l: max(l) / len(l),
        }
        self.model = predictors[predictor_type]
        self.backup_model = predictors[backup_predictor_type]
        self.backup_predictor_duration = backup_predictor_duration


    def predict(self, series: List[int]):
        series_aggregated = []
        step = 10  # take maximum of each past 10 seconds
        for i in range(0, len(series), step):
            series_aggregated.append(max(series[i : i + step]))
        if len(series_aggregated) >= int((self.backup_predictor_duration * 60) / step):
            if self.predictor_type == "lstm":
                if len(series_aggregated) < LSTM_INPUT_SIZE:
                    # corner case of bigger output from prometheus
                    series_aggregated = series_aggregated[-LSTM_INPUT_SIZE:]
                    logger.info(
                        "not enough information for lstm"
                        f" using backup predictor {self.backup_predictor}"
                    )
                    return self.backup_model(series_aggregated)
                model_intput = tf.convert_to_tensor(
                    np.array(series_aggregated).reshape((-1, LSTM_INPUT_SIZE, 1)),
                    dtype=tf.float32,
                )
                model_output = self.model.predict(model_intput)[0][0]
            else:
                model_output = self.model(series)
        else:
            model_output = self.backup_model(series)
        return model_output
