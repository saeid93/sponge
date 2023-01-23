from typing import Dict, List
import numpy as np
import pandas as pd
import itertools
from sklearn import linear_model

class ResourceAllocation:
    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        # For now only one type CPU/GPU allocation is allowed
        if cpu != 0 and gpu != 0:
            raise ValueError(
                'For now only one of the CPU or GPU allocation is allowed')
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
        # TODO decide whether latency based or RPS
        throughput = (1/self.latency) * self.batch
        return throughput

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
        resource_allocation: ResourceAllocation,
        measured_profiles: List[Profile],
        accuracy: float) -> None:

        self.resource_allocation = resource_allocation
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
        available_model_profiles: List[Model], # TODO not dict and search instead
        active_variant: str, active_allocation: ResourceAllocation,
        replica: int, batch: int,
        gpu_mode: False) -> None:
        self.available_model_profiles = available_model_profiles
        self.active_variant = active_variant
        self.active_allocation = active_allocation
        self.replicas = replica
        self.batch = batch
        self.replicas = replica
        self.gpu_mode = gpu_mode
        self.name = name
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.gpu_mode:
                    if self.active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        break
                else:
                    if self.active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        break
        else: # no-break
            raise ValueError(f"no matching profile for the variant {active_variant} and allocation"
                             f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}")

    def model_switch(self, active_variant: str) -> None:
        """
        changes variant under specific allocation
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.gpu_mode:
                    if self.active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        self.active_variant = active_variant
                        break
                else:
                    if self.active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        self.active_variant = active_variant
                        break
        else: # no-break
            raise ValueError(f"no matching profile for the variant {active_variant} and allocation"
                             f"of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}")

    def change_allocation(self, active_allocation: ResourceAllocation) -> None:
        """
        change allocation of a specific variant
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == self.active_variant:
                if self.gpu_mode:
                    if active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        self.active_allocation = active_allocation
                        break
                else:
                    if active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        self.active_allocation = active_allocation
                        break
        else: # no-break
            raise ValueError(f"no matching profile for the variant {self.active_variant} and allocation"
                             f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}")

    def re_scale(self, replica) -> None:
        self.replicas = replica

    def change_batch(self, batch) -> None:
        self.batch = batch

    @property
    def active_model(self) -> Model:
        return self.available_model_profiles[self.active_variant_index]

    @property
    def cpu(self) -> int:
        if self.gpu_mode:
            raise ValueError('The node is on gpu mode')
        else:
            return self.active_model.resource_allocation.cpu

    @property
    def gpu(self) -> float:
        if self.gpu_mode:
            return self.active_model.resource_allocation.gpu
        else:
            return 0

    @property
    def cpu_all_replicas(self) -> int:
        if self.gpu_mode:
            raise ValueError('The node is on gpu mode')
        else:
            return self.active_model.resource_allocation.cpu * self.replicas

    @property
    def gpu_all_replicas(self) -> float:
        if self.gpu_mode:
            return self.active_model.resource_allocation.gpu * self.replicas
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
        return list(set(map(
            lambda l: l.name, self.available_model_profiles)))

    @property
    def batches(self):
        batches = list(map(
            lambda l: l.batch, self.active_model.profiles))
        return batches

    @property
    def resource_allocations_cpu_mode(self):
        cpu_allocations = list(set(
            list(map(
                lambda l: l.resource_allocation.cpu,
                self.available_model_profiles))))
        resource_allocations = list(map(
            lambda l: ResourceAllocation(cpu=l), cpu_allocations))
        return resource_allocations

    @property
    def resource_allocations_gpu_mode(self):
        gpu_allocations = list(set(
            list(map(
                lambda l: l.resource_allocation.gpu,
                self.available_model_profiles))))
        resource_allocations = list(map(
            lambda l: ResourceAllocation(gpu=l), gpu_allocations))
        return resource_allocations

class Pipeline:
    def __init__(
        self,
        inference_graph: List[Task], gpu_mode: bool) -> None:
        self.inference_graph = inference_graph
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

    def accuracy_objective(self) -> float:
        """
        objective function of the pipeline
        """
        accuracy_objective = self.pipeline.pipeline_accuracy
        return accuracy_objective

    def resource_objective(self) -> float:
        """
        objective function of the pipeline
        """
        resource_objective = 1/self.pipeline.cpu_usage
        return resource_objective

    def objective(
        self,
        alpha: float, beta: float) -> None:
        """
        objective function of the pipeline
        """
        # TODO make it reconfigurable
        objective = alpha * self.accuracy_objective() +\
            beta * self.resource_objective()
        return objective

    def constraints(self, arrival_rate: int, sla: float) -> bool:
        """
        whether the constraints are met or not
        """
        if self.sla_is_met(sla=sla) and self.can_sustain_load(
            arrival_rate=arrival_rate):
            return True
        return False

    def all_states(
        self,
        check_constraints: bool = False,
        scaling_cap: int = 2,
        alpha: float = 1,
        beta: float = 1,
        arrival_rate: int = None,
        sla: float = None) -> pd.DataFrame:
        """
        scaling_cap: maximum number of allowed replication
        returns all states of the pipeline
        """
        variant_names = []
        replicas = []
        batches = []
        allocations = []
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            replicas.append(np.arange(1, scaling_cap+1))
            batches.append(task.batches)
            if task.gpu_mode:
                allocations.append(task.resource_allocations_gpu_mode)
            else:
                allocations.append(task.resource_allocations_cpu_mode)
        variant_names = list(itertools.product(*variant_names))
        replicas = list(itertools.product(*replicas))
        batches = list(itertools.product(*batches))
        allocations = list(itertools.product(*allocations))

        states = self._generate_states()
        all_combinations = list(
            itertools.product(*[variant_names, allocations, replicas, batches]))
        for combination in all_combinations:
            for task_id_i in range(self.pipeline.num_nodes):
                # change config knobs (model_variant, batch, scale)
                self.pipeline.inference_graph[
                    task_id_i].model_switch(
                        active_variant=combination[0][task_id_i])
                self.pipeline.inference_graph[
                    task_id_i].change_allocation(
                        active_allocation=combination[1][task_id_i])
                self.pipeline.inference_graph[
                    task_id_i].re_scale(replica=combination[2][task_id_i])
                self.pipeline.inference_graph[
                    task_id_i].change_batch(batch=combination[3][task_id_i])
            ok_to_add = False
            if check_constraints:
                if self.constraints(arrival_rate=arrival_rate, sla=sla):
                    ok_to_add = True
            else:
                ok_to_add = True
            if ok_to_add:
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
                    state['accuracy_objective'] =\
                        self.accuracy_objective()
                    state['resource_objective'] =\
                        self.resource_objective()
                    state['objective'] = self.objective(alpha=alpha, beta=beta)
                    state['alpha'] = alpha
                    state['beta'] = beta
                states = states.append(state, ignore_index=True)
        return states

    def greedy_optimizer(self, scaling_cap: int, sla: float, arrival_rate: int):
        states = self.all_states(
            scaling_cap=scaling_cap, sla=sla, arrival_rate=arrival_rate)
        optimal = states[states['objective'] == states['objective'].max()]
        return optimal

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

    def sla_is_met(self, sla) -> bool:
        return self.pipeline.pipeline_latency < sla


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
