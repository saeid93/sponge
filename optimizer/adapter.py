from typing import Dict, Literal, Tuple
import time
from kubernetes import config
from kubernetes import client
from typing import List
import time
import os
import sys
import pandas as pd
import concurrent.futures

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..')))
from experiments.utils.pipeline_operations import (
    check_node_up
)
from experiments.utils.prometheus import PromClient
prom_client = PromClient()

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

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
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..', '..')))

from optimizer import (
    Optimizer,
    Pipeline
)
from experiments.utils.constants import (
    NAMESPACE
)
from optimizer.optimizer import Optimizer

class Adapter:
    def __init__(
            self,
            pipeline_name: str,
            pipeline: Pipeline,
            node_names: List[str],
            adaptation_interval: int,
            optimization_method: Literal['gurobi', 'brute-force'],
            allocation_mode: Literal['base', 'variable'],
            only_measured_profiles: bool,
            scaling_cap: int,
            alpha: float,
            beta: float,
            gamma: float,
            num_state_limit: int,
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
            random_sample=False
        )
        self.optimization_method = optimization_method
        self.scaling_cap = scaling_cap
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_state_limit = num_state_limit
        self.monitoring = Monitoring(
            pipeline_name=self.pipeline_name)
        self.predictor = Predictor()

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
        pipeline_up = False
        pipeline_up = check_node_up(node_name='router', silent_mode=True)

        while True:
            time.sleep(self.adaptation_interval)
            rps_series = self.monitoring.monitor()
            predicted_load = self.predictor.predict(rps_series)
            optimal = self.optimizer.optimize(
                optimization_method=self.optimization_method,
                scaling_cap=self.scaling_cap,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                arrival_rate=predicted_load,
                num_state_limit=self.num_state_limit
            )
            new_configs = self.output_parser(optimal)
            to_apply_config = self.choose_config(new_configs)
            self.change_pipeline_config(to_apply_config)
            pipeline_up = check_node_up(node_name='router', silent_mode=True)
            if not pipeline_up:
                print('no pipeline in the system, aborting adaptation process ...')
                # with the message that the process has ended
                break

    def output_parser(self, optimizer_output: pd.DataFrame):
        new_configs = []
        for _, row in optimizer_output.iterrows():
            config = {}
            for task_id, task_name in enumerate(self.node_names):
                config[task_name] = {}
                # config[task_name]['cpu'] = row[f'task_{task_id}_cpu']
                config[task_name]['replicas'] = int(row[f'task_{task_id}_replicas'])
                config[task_name]['batch'] = int(row[f'task_{task_id}_batch'])
                config[task_name]['variant'] = row[f'task_{task_id}_variant']
            new_configs.append(config)
        return new_configs

    def choose_config(self, new_configs):
        # TODO TEMP workaround
        # This should be from comparing with the
        # current config
        # easiest for now is to choose config with
        # with the least change from former config
        current_config = self.extract_config()
        chosen_config = new_configs[-1]
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
                name=node_name)
            component_config = raw_config['spec']['predictors'][0]['componentSpecs'][0]
            env_vars = component_config['spec']['containers'][0]['env']
            replicas = component_config['replicas']
            cpu = int(component_config[
                'spec']['containers'][0]['resources']['requests']['cpu'])
            for env_var in env_vars:
                if env_var['name'] == 'MODEL_VARIANT':
                    variant = env_var['value']
                if env_var['name'] == 'MLSERVER_MODEL_MAX_BATCH_SIZE':
                    batch = env_var['value']
            node_config['replicas'] = replicas
            node_config['variant'] = variant
            node_config['batch'] = batch
            # node_config['cpu'] = cpu
            current_config[node_name] = node_config

        return current_config

    def change_pipeline_config(
            self,
            config: Dict[str, Dict[str, int]]):
        """change the existing configuration based on the optimizer
           output
        Args:
            config (Dict[str, Dict[str, int]]): _description_
        """
        node_names = list(config.keys())
        node_configs = list(config.values())
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self.change_node_config, zip(node_names, node_configs)))
        return results

    def change_node_config(
            self,
            inputs: Tuple[str, Dict[str, int]]):
        node_name, node_config = inputs
        deployment_config = kube_custom_api.get_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=NAMESPACE,
            plural="seldondeployments",
            name=node_name)
        deployment_config['spec'][
            'predictors'][0]['componentSpecs'][0]['replicas'] = node_config['replicas']
        for env_index, env_var in enumerate(deployment_config['spec'][
            'predictors'][0][
            'componentSpecs'][0]['spec']['containers'][0]['env']):
            if env_var['name'] == 'MODEL_VARIANT':
                deployment_config['spec'][
                                'predictors'][0][
                                'componentSpecs'][0][
                    'spec']['containers'][0]['env'][env_index]['value'] =\
                        node_config['variant']
            if env_var['name'] == 'MLSERVER_MODEL_MAX_BATCH_SIZE':
                deployment_config['spec'][
                                'predictors'][0][
                                'componentSpecs'][0][
                    'spec']['containers'][0]['env'][env_index]['value'] =\
                        str(node_config['batch'])
        kube_custom_api.replace_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=NAMESPACE,
            plural="seldondeployments",
            name=node_name,
            body=deployment_config)
        return True

class Monitoring:
    def __init__(self, pipeline_name) -> None:
        self.pipeline_name = pipeline_name
    # Get the rps of the router
    def monitor(self) -> List[int]:
        duration = 1
        rate = 15
        rps_series, _ = prom_client.get_request_per_second(
            pod_name='router', namespace="default",
            duration=duration, container='router', rate=15)
        return rps_series
    # if needed to enquire load from multiple
    # replica then simply sum them


class Predictor:
    def __init__(self) -> None:
        # TODO add the lstm
        self.model = lambda l: l[-1]
    
    def predict(self, series: List[int]):
        return self.model(series) # TEMP

