from typing import Dict, List
import numpy as np
import pandas as pd
import itertools
from sklearn import linear_model

class ResourceRequirement:
    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        self.cpu = cpu
        self.gpu = gpu

class Profile:
    def __init__(
        self, batch: int, latency: float, measured: bool = True) -> None:
        self.batch = batch
        self.latency = latency
        self.measured = measured
    
    @property
    def throughput(self):
        return 1/self.latency

    def __eq__(self, other):
        if not isinstance(other, int):
            raise TypeError("batch size variables should be int")
        if other == self.batch:
            return True
        return False

class Model:
    def __init__(
        self,
        name: str,
        resources: ResourceRequirement,
        measured_profiles: List[Profile],
        accuracy: float) -> None:

        self.resources = resources
        self.measured_profiles = measured_profiles
        self.accuracy = accuracy
        self.name = name
        self.profiles = self.regression_model()

    def regression_model(self) -> List[Profile]:
        """
        interapolate the latency for unknown batch sizes
        """
        train_x = np.array(list(map(
            lambda l: l.batch, self.measured_profiles))).reshape(-1, 1)
        train_y = np.array(list(map(
            lambda l: l.latency, self.measured_profiles))).reshape(-1, 1)
        all_x = np.arange(self.min_batch, self.max_batch+1)
        test_x = all_x[~np.isin(all_x, train_x)].reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(train_x, train_y)
        test_y = regr.predict(test_x)
        predicted_profiles = []
        for x, y in zip(test_x.reshape(-1), test_y.reshape(-1)):
            predicted_profiles.append(
                Profile(batch=x, latency=y, measured=False))
        profiles: List[Profile] = predicted_profiles + self.measured_profiles
        profiles.sort(key=lambda profile: profile.batch)
        return profiles

    @property
    def profiled_batches(self):
        batches = [profile.batch for profile in self.measured_profiles]
        return batches

    @property
    def min_batch(self):
        return min(self.profiled_batches)

    @property
    def max_batch(self):
        return max(self.profiled_batches)

class Task:
    def __init__(
        self,
        name: str,
        available_variants: Dict[str, Model],
        active_variant: str, replica: int, batch: int,
        gpu_mode: False) -> None:
        self.available_variants = available_variants
        self.replicas = replica
        self.batch = batch
        if active_variant not in self.available_variants.keys():
            raise ValueError(f"{active_variant} not part of variants",
                             f"available variants: {self.available_variants}")
        self.active_variant = active_variant
        self.replicas = replica
        self.gpu_mode = gpu_mode
        self.name = name
    # TODO add queue delay

    def model_switch(self, active_variant) -> None:
        if active_variant not in self.available_variants.keys():
            raise ValueError(f"{active_variant} not part of variants",
                             f"available variants: {self.available_variants}")
        self.active_variant = active_variant

    def re_scale(self, replica) -> None:
        self.replicas = replica

    def change_batch(self, batch) -> None:
        self.batch = batch

    @property
    def active_model(self) -> Model:
        return self.available_variants[self.active_variant]

    @property
    def cpu(self) -> int:
        if self.gpu_mode:
            raise ValueError('The node is on gpu mode')
        else:
            return self.active_model.resources.cpu

    @property
    def gpu(self) -> float:
        if self.gpu_mode:
            return self.active_model.resources.gpu
        else:
            return 0

    @property
    def cpu_all_replicas(self) -> int:
        if self.gpu_mode:
            raise ValueError('The node is on gpu mode')
        else:
            return self.active_model.resources.cpu * self.replicas

    @property
    def gpu_all_replicas(self) -> float:
        if self.gpu_mode:
            return self.active_model.resources.gpu * self.replicas
        return 0

    @property
    def latency(self) -> float:
        latency = next(filter(
            lambda profile: profile.batch == self.batch,
            self.active_model.profiles)).latency
        return latency

    @property
    def throughput(self) -> float:
        throughput = next(filter(
            lambda profile: profile.batch == self.batch,
            self.active_model.profiles)).throughput
        return throughput

    @property
    def measured(self) -> bool:
        measured = next(filter(
            lambda profile: profile.batch == self.batch,
            self.active_model.profiles)).measured
        return measured

    @property
    def throughput_all_replicas(self):
        return self.throughput * self.replicas

    @property
    def accuracy(self):
        return self.active_model.accuracy

    @property
    def variant_names(self):
        return list(self.available_variants.keys())

    @property
    def batches(self):
        batches = list(map(
            lambda l: l.batch, self.active_model.profiles))
        return batches

class Pipeline:
    def __init__(
        self, sla: float,
        inference_graph: List[Task], gpu_mode: bool) -> None:
        self.inference_graph = inference_graph
        self.sla = sla
        self.gpu_mode = gpu_mode
        if not self.gpu_mode:
            for task in self.inference_graph:
                if task.gpu_mode:
                    raise ValueError(
                        f'pipeline is deployed on cpu',
                        f'but task {task.name} is on gpu')

    @property
    def stage_wise_throughput(self):
        throughputs = list(
            map(lambda l: l.throughput_all_replicas, self.inference_graph))
        return throughputs

    @property
    def stage_wise_latencies(self):
        latencies = list(
            map(lambda l: l.latency, self.inference_graph))
        return latencies

    @property
    def stage_wise_accuracies(self):
        latencies = list(
            map(lambda l: l.accuracy, self.inference_graph))
        return latencies

    @property
    def stage_wise_replicas(self):
        replicas = list(
            map(lambda l: l.replicas, self.inference_graph))
        return replicas

    @property
    def stage_wise_cpu(self):
        cpu = []
        for task in self.inference_graph:
            if not task.gpu_mode:
                cpu.append(task.cpu_all_replicas)
            else:
                cpu.append(0)
        return cpu

    @property
    def stage_wise_gpu(self):
        gpu = []
        for task in self.inference_graph:
            if task.gpu_mode:
                gpu.append(task.gpu_all_replicas)
            else:
                gpu.append(0)
        return gpu

    @property
    def pipeline_cpu(self):
        return sum(self.stage_wise_cpu)

    @property
    def pipeline_gpu(self):
        return sum(self.stage_wise_gpu)

    @property    
    def pipeline_latency(self):
        return sum(self.stage_wise_latencies)

    @property    
    def pipeline_accuracy(self):
        accuracy = 1
        for task in self.inference_graph:
            accuracy *= task.accuracy
        return accuracy

    @property    
    def pipeline_throughput(self):
        return min(self.stage_wise_throughput)

    @property
    def cpu_usage(self):
        return sum(self.stage_wise_cpu)

    @property
    def gpu_usage(self):
        return sum(self.stage_wise_gpu)

    @property
    def num_nodes(self):
        return len(self.inference_graph)

    def visualize(self):
        pass

class Optimizer:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.headers = self._generate_states_headers()
        # self.states = self._generate_states()

    def _generate_states_headers(self):
        per_node_header_templates = [
            'variant', 'cpu', 'gpu', 'cpu_all_replicas',
            'gpu_all_replicas', 'batch', 'replicas',
            'latency', 'throughput', 'throughput_all_replicas',
            'measured']
        headers = []
        for task_id in range(self.pipeline.num_nodes):
            for header in per_node_header_templates:
                headers.append(f"task_{task_id}_{header}")
        e2e_headers = [
            'pipeline_latency', 'pipeline_throughput',
            'pipeline_cpu', 'pipeline_gpu']
        headers += e2e_headers
        return headers

    def _generate_states(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.headers)

    def objective(
        self,
        alpha: float, beta: float) -> None:
        objective = alpha * self.pipeline.pipeline_accuracy +\
            beta * 1/self.pipeline.cpu_usage
        return objective

    def all_states(self, scaling_cap: 10) -> pd.DataFrame:
        """
        scaling_cap: maximum number of allowed replication
        returns all states of the pipeline
        """
        variant_names = []
        # model_variants = []
        replicas = []
        batches = []
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            # model_variants.append(task.available_variants.values())
            replicas.append(np.arange(1, scaling_cap+1))
            batches.append(task.batches)
        variant_names = list(itertools.product(*variant_names))
        # model_variants = list(itertools.product(*model_variants))
        replicas = list(itertools.product(*replicas))
        batches = list(itertools.product(*batches))

        states = self._generate_states()
        all_combinations = list(
            itertools.product(*[variant_names, replicas, batches]))
        for combination in all_combinations:
            for task_id_i in range(self.pipeline.num_nodes):
                # change config knobs (model_variant, batch, scale)
                self.pipeline.inference_graph[
                    task_id_i].model_switch(
                        active_variant=combination[0][task_id_i])
                self.pipeline.inference_graph[
                    task_id_i].re_scale(replica=combination[1][task_id_i])
                self.pipeline.inference_graph[
                    task_id_i].change_batch(batch=combination[2][task_id_i])
            state = {}
            for task_id_j in range(self.pipeline.num_nodes):
                # record all stats under this configs
                state[f'task_{task_id_j}_variant'] =\
                    self.pipeline.inference_graph[task_id_j].active_variant
                state[f'task_{task_id_j}_cpu'] =\
                    self.pipeline.inference_graph[task_id_j].cpu
                state[f'task_{task_id_j}_gpu'] =\
                    self.pipeline.inference_graph[task_id_j].gpu
                state[f'task_{task_id_j}_cpu_all_replicas'] =\
                    self.pipeline.inference_graph[task_id_j].cpu_all_replicas
                state[f'task_{task_id_j}_gpu_all_replicas'] =\
                    self.pipeline.inference_graph[task_id_j].gpu_all_replicas
                state[f'task_{task_id_j}_batch'] =\
                    self.pipeline.inference_graph[task_id_j].batch
                state[f'task_{task_id_j}_replicas'] =\
                    self.pipeline.inference_graph[task_id_j].replicas
                state[f'task_{task_id_j}_latency'] =\
                    self.pipeline.inference_graph[task_id_j].latency
                state[f'task_{task_id_j}_throughput'] =\
                    self.pipeline.inference_graph[task_id_j].throughput
                state[f'task_{task_id_j}_throughput_all_replicas'] =\
                    self.pipeline.inference_graph[task_id_j].throughput_all_replicas
                state[f'task_{task_id_j}_accuracy'] =\
                    self.pipeline.inference_graph[task_id_j].accuracy
                state[f'task_{task_id_j}_measured'] =\
                    self.pipeline.inference_graph[task_id_j].measured
                state['pipeline_accuracy'] =\
                    self.pipeline.pipeline_accuracy
                state['pipeline_latency'] =\
                    self.pipeline.pipeline_latency
                state['pipeline_throughput'] =\
                    self.pipeline.pipeline_throughput
                state['pipeline_cpu'] =\
                    self.pipeline.pipeline_cpu
                state['pipeline_gpu'] =\
                    self.pipeline.pipeline_gpu
            states = states.append(state, ignore_index=True)
        return states

    def all_feasible_states(self):
        pass

    def greedy_optimizer(self):
        pass

    def can_sustain_load(
        self,
        arrival_rate: int) -> bool:
        """
        whether the existing config can sustain a load
        """
        for task in self.pipeline.inference_graph:
            if arrival_rate > task.throughput_all_replicas:
                return False
        return True

    def find_load_bottlenecks(
        self,
        arrival_rate: int) -> List[int]:
        """
        whether the existing config can sustain a load
        """
        if self.can_sustain_load(
            arrival_rate=arrival_rate):
            raise ValueError(f'The load can be sustained! no bottleneck!')
        bottlenecks = []
        for task_id, task in enumerate(self.pipeline.inference_graph):
            if arrival_rate > task.throughput_all_replicas:
                bottlenecks.append(task_id)
        return bottlenecks

    def sla_is_met(self) -> bool:
        return self.pipeline.pipeline_latency < self.pipeline.sla
