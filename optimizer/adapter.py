from typing import Dict, Literal
import time
from kubernetes import config
from kubernetes import client
import time
import os
import sys
import pandas as pd

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)
kube_api = client.api.core_v1_api.CoreV1Api()

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..', '..')))

from optimizer import (
    Optimizer
)
from experiments.utils.constants import (
    NAMESPACE
)
from optimizer.optimizer import Optimizer


class Adapter:
    def __init__(
            self,
            pipeline_name: str,
            adaptation_interval: int,
            optimization_method: Literal['gurobi', 'brute-force'],
            allocation_mode: Literal['base', 'variable'],
            only_measured_profiles: bool,
            scaling_cap: int,
            alpha: float,
            beta: float,
            gamma: float,
            num_state_limit: int) -> None:
        self.pipeline_name = pipeline_name
        self.adaptation_interval = adaptation_interval
        self.lstm = lambda l: l[-1] # TEMP TODO replace with real lstm
        self.optimizer = Optimizer(
            pipeline=pipeline_name,
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
    def start(self):
        no_pipeline = False
        while True:
            time.sleep(self.adaptation_interval)
            rps_series = self.monitoring.monitor()
            predicted_load = self.lstm(rps_series)
            self.optimizer(
                optimization_method=self.optimization_method,
                scaling_cap=self.scaling_cap,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                arrival_rate=predicted_load,
                num_state_limit=self.num_state_limit
            )
            if no_pipeline:
                # TODO check if there is not a pipeline stop wthile
                # with the message that the process has ended
                break


        # 1. Have the optimizer object of the pipelien here
        # 2. For now and Adapter per pipeline -> an adapter for entire cluster that assigne each pipeline to it
        # 3. TODO add LSTM for predicting the load
        # 4. Use monitoring for periodically checking the status of the pipeline in terms of load (and maybe other metrics)
        # 5. Give the load and pipeline status to the optimizer
        # 6. Find the best answer
        # 7. Use the change config script to change the pipelien to the new config
    def output_parser(self, optimizer_output: pd.DataFrame):
        pass


    def change_config(
            self,
            config: Dict[str, Dict[str, int]]):
        """change the existing configuration based on the optimizer
           output
        Args:
            config (Dict[str, Dict[str, int]]): _description_
        """


        from kubernetes import client, config

        deployment_name = "my-seldon-deployment"
        namespace = "default"

        config.load_kube_config()
        api_instance = client.CustomObjectsApi()

        seldon_deployment = api_instance.get_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=namespace,
            plural="seldondeployments",
            name=deployment_name
        )
        seldon_deployment["spec"]["predictors"][0]["replicas"] = 3
        api_instance.replace_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=namespace,
            plural="seldondeployments",
            name=deployment_name,
            body=seldon_deployment
        )
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


class Monitoring:
    def __init__(self, pipeline_name) -> None:
        self.pipeline_name = pipeline_name
    # TODO get rps recieved at the first model, I think this will be entrance
    def monitor(self):
        rps_series = [10, 10, 10]
        return rps_series
    # if needed to enquire load from multiple
    # replica then simply sum them
