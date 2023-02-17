from typing import Dict, List, Any
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import itertools
from sklearn import linear_model
from copy import deepcopy

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
        self.accuracy = accuracy/100
        self.normalized_accuracy = None
        self.name = name
        self.profiles, self.latency_model_params = self.regression_model()

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
        latency_model = linear_model.LinearRegression()
        latency_model.fit(train_x, train_y)
        test_y = latency_model.predict(test_x)
        predicted_profiles = []
        for x, y in zip(test_x.reshape(-1), test_y.reshape(-1)):
            predicted_profiles.append(
                Profile(batch=x, latency=y, measured=False))
        profiles: List[Profile] = predicted_profiles + self.measured_profiles
        profiles.sort(key=lambda profile: profile.batch)
        return profiles, [latency_model.coef_[0][0], latency_model.intercept_[0]]

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

    @property
    def max_batch(self):
        return max(self.profiled_batches)

class Task:
    def __init__(
        self,
        name: str,
        available_model_profiles: List[Model],
        active_variant: str,
        active_allocation: ResourceAllocation,
        replica: int, batch: int,
        allocation_mode: str,
        threshold: int,
        sla_factor: int,
        normalize_accuracy: bool,
        gpu_mode: False) -> None:

        self.available_model_profiles = available_model_profiles
        self.active_variant = active_variant
        self.active_allocation = active_allocation
        self.initial_allocation = active_allocation
        self.replicas = replica
        self.batch = batch
        self.replicas = replica
        self.gpu_mode = gpu_mode
        self.normalize_accuracy = normalize_accuracy
        self.threshold = threshold
        self.name = name
        self.sla_factor = sla_factor
        self.allocation_mode = allocation_mode
        self.sla = self.find_task_sla()
        self.variants_accuracies = self.find_variants_accuracies()
        self.variants_accuracies_normalized =\
            self.find_variants_accuracies_normalized()
        if allocation_mode == 'base':
            self.base_allocations = self.find_base_allocation()

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
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
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
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f"of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}")

        if self.allocation_mode == 'base':
            self.set_to_base_allocation()

    def find_task_sla(self) -> Dict[str, ResourceAllocation]:
        models = {key: [] for key in self.variant_names}
        # 1. filter out models
        for model_variant in self.variant_names:
            for allocation in self.available_model_profiles:
                if allocation.name == model_variant:
                    models[model_variant].append(allocation)
        # 2. find variant SLA
        model_slas = {}
        for model, allocation in models.items():
            # finding sla of each model
            # sla is latency of minimum batch
            # under minimum resource multiplied by
            # a given scaling factor
            # since allocations are sorted the first
            # one will be the one with maximum resource req
            sla = allocation[-1].profiles[0].latency * self.sla_factor
            model_slas[model] = sla
        task_sla = min(model_slas.values())
        return task_sla

    def find_base_allocation(self) -> Dict[str, ResourceAllocation]:
        models = {key: [] for key in self.variant_names}
        # 1. filter out models
        for model_variant in self.variant_names:
            for allocation in self.available_model_profiles:
                if allocation.name == model_variant:
                    models[model_variant].append(allocation)
        base_allocation = {}
        for model_variant, allocations in models.items():
            # finding the mimumu allocation that can respond
            # to the threshold
            # the profiles are sorted therefore therefore
            # we iterate from the first profile
            for allocation in allocations:
                # check if the max batch size throughput
                # can reponsd to the threshold
                if allocation.profiles[-1].throughput >= self.threshold\
                    and allocation.profiles[-1].throughput >= self.sla:
                    base_allocation[model_variant] = deepcopy(
                        allocation.resource_allocation)
                    break
            else: # no-break
                raise ValueError(
                    f'No responsive model profile to threshold {self.threshold}'
                    f' or model sla {self.sla} was found'
                    f' for model variant {model_variant}'
                    'consider either changing the the threshold or '
                    f'sla factor {self.sla_factor}')
        return base_allocation

    def set_to_base_allocation(self):
        self.change_allocation(
            active_allocation=self.base_allocations[self.active_variant])

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
            raise ValueError(
                f"no matching profile for the variant {self.active_variant} and allocation"
                f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}")

    def re_scale(self, replica) -> None:
        self.replicas = replica

    def change_batch(self, batch) -> None:
        self.batch = batch

    def find_variants_accuracies(self) -> Dict[str, float]:
        """create all the accuracies for each task

        Returns:
            Dict[str, float]: variant accuracies
        """
        variants_accuracies = {}
        for profile in self.available_model_profiles:
            variants_accuracies[profile.name] = profile.accuracy
        return variants_accuracies

    def find_variants_accuracies_normalized(self) -> Dict[str, float]:
        """create normalized accuracies for each task

        Returns:
            Dict[str, float]: varaint accuracies
        """
        variants = []
        accuracies = []
        for variant, accuracy in self.variants_accuracies.items():
            variants.append(variant)
            accuracies.append(accuracy)
        variants = [variant for _, variant in sorted(zip(accuracies, variants))]
        accuracies.sort()
        accuracies_normalized = (np.arange(
            len(accuracies))/(len(accuracies)-1)).tolist()
        variants_accuracies_normalized = {
            variant: accuracy_normalized for variant, accuracy_normalized in zip(
                variants, accuracies_normalized)}
        return variants_accuracies_normalized 

    @property
    def active_model(self) -> Model:
        return self.available_model_profiles[self.active_variant_index]

    @property
    def latency_model_params(self) -> Model:
        return self.available_model_profiles[self.active_variant_index].latency_model_params

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
    def queue_latency_params(self) -> float:
        # TODO add a function to infer queue latency
        queue_latency_params = 0
        return [queue_latency_params]

    @property
    def queue_latency(self) -> float:
        # TODO TEMP
        queue_latency = 0
        return queue_latency

    @property
    def model_latency(self) -> float:
        latency = next(filter(
            lambda profile: profile.batch == self.batch,
            self.active_model.profiles)).latency
        return latency

    @property
    def latency(self) -> float:
        latency = self.model_latency + self.queue_latency
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
        if self.normalize_accuracy:
            return self.variants_accuracies_normalized[
                self.active_variant]
        else:
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
        inference_graph: List[Task],
        gpu_mode: bool,
        sla_factor: int,
        accuracy_method: str,
        normalize_accuracy: bool
        ) -> None:
        self.inference_graph: List[Task] = inference_graph
        self.gpu_mode = gpu_mode
        self.sla_factor = sla_factor
        self.accuracy_method = accuracy_method
        self.normalize_accuracy = normalize_accuracy
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
    def sla(self):
        sla = sum(
            map(lambda l: l.sla, self.inference_graph))
        return sla

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
    def stage_wise_task_names(self):
        task_names = []
        for task in self.inference_graph:
            task_names.append(task.name)
        return task_names

    @property
    def stage_wise_available_variants(self):
        task_names = {}
        for task in self.inference_graph:
            task_names[task.name] = task.variant_names
        return task_names

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
        tasks_accuracies = {}
        for task in self.inference_graph:
            acive_variant = task.active_variant
            if self.normalize_accuracy:
                accuracy =\
                    task.variants_accuracies_normalized[acive_variant]
            else:
                accuracy =\
                    task.variants_accuracies[acive_variant]
            tasks_accuracies[acive_variant] = accuracy
        if self.accuracy_method == 'multiply':
            accuracy = 1
            for task, task_accuracy in tasks_accuracies.items():
                accuracy *= task_accuracy
        elif self.accuracy_method == 'sum':
            accuracy = 0
            for task, task_accuracy in tasks_accuracies.items():
                accuracy += task_accuracy
        elif self.accuracy_method == 'average':
            accuracy = 0
            for task, task_accuracy in tasks_accuracies.items():
                accuracy += task_accuracy
            accuracy /= len(self.inference_graph)
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
    def __init__(
        self, pipeline: Pipeline,
        allocation_mode: str) -> None:
        self.pipeline = pipeline
        self.allocation_mode = allocation_mode
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
        resource_objective = self.pipeline.cpu_usage
        return resource_objective

    def batch_objective(self) -> float:
        """
        batch objecive of the pipeline
        """
        max_batch = 0
        for task in self.pipeline.inference_graph:
            max_batch += task.batch
        return max_batch

    def objective(
        self,
        alpha: float, beta: float, gamma: float) -> None:
        """
        objective function of the pipeline
        """
        objective = alpha * self.accuracy_objective() -\
            beta * self.resource_objective() +\
                gamma * self.batch_objective()
        return objective

    def constraints(self, arrival_rate: int, sla: float) -> bool:
        """
        whether the constraints are met or not
        """
        if self.sla_is_met(sla=sla) and self.can_sustain_load(
            arrival_rate=arrival_rate):
            return True
        return False

    def pipeline_latency_upper_bound(self) -> float:
        # maximum number for latency of a node in
        # a pipeline
        max_model_latencies = 0
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            for variant_name in task.variant_names:
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    task.change_batch(batch_size)
                    if task.model_latency > max_model_latencies:
                        max_model_latencies = task.model_latency
        return max_model_latencies

    def latency_parameters(self) -> Dict[str, Dict[str, List[float]]]:
        # latencies of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        model_latencies = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            model_latencies[task.name] = {}
            for variant_name in task.variant_names:
                model_latencies[task.name][variant_name] = {}
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    task.change_batch(batch_size)
                    model_latencies[
                        task.name][variant_name] = task.latency_model_params
        return model_latencies

    def accuracy_parameters(self) -> Dict[str, Dict[str, float]]:
        # accuracies of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        model_accuracies = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            model_accuracies[task.name] = {}
            for variant_name in task.variant_names:
                model_accuracies[task.name][variant_name] = {}
                task.model_switch(variant_name)
                model_accuracies[
                    task.name][variant_name] = task.accuracy
        return model_accuracies

    def queue_parameters(self) -> Dict[str, float]:
        # queue latencies of all cases nested dictionary
        # for gorubi solver
        # [stage_name]
        queue_latencies = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            queue_latencies[task.name] = {}
            for batch_size in task.batches:
                task.change_batch(batch_size)
                queue_latencies[task.name] = task.queue_latency_params
        return queue_latencies


    def base_allocations(self):
        # base allocation of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        base_allocations = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            if self.pipeline.gpu_mode:
                base_allocations[task.name] = {
                    key: value.gpu for (
                        key, value) in task.base_allocations.items()}
            else:
                base_allocations[task.name] = {
                    key: value.cpu for (
                        key, value) in task.base_allocations.items()}
        return base_allocations

    def all_states(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        check_constraints: bool,
        arrival_rate: int,
        num_state_limit: int = None,
        ) -> pd.DataFrame:
        """generate all the possible states based on profiling data

        Args:
            check_constraints (bool, optional): whether to check the
                objective function contraint or not. Defaults to False.
            scaling_cap (int, optional): maximum number of allowed horizontal
                scaling for each node. Defaults to 2.
            alpha (float, optional): accuracy ojbective weight.
                Defaults to 1.
            beta (float, optional): resource usage
                objective weigth. Defaults to 1.
            gamma (float, optional): batch size
                objective batch. Defaults to 1.
            arrival_rate (int, optional): arrival rate into
                the pipeline. Defaults to None.
            state_limit (int, optional): whether to generate a
                fixed number of state. Defaults to None.

        Returns:
            pd.DataFrame: all the states of the pipeline
        """
        sla = self.pipeline.sla
        if num_state_limit is not None:
            state_counter = 0
        variant_names = []
        replicas = []
        batches = []
        allocations = []
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            replicas.append(np.arange(1, scaling_cap+1))
            batches.append(task.batches)
            if self.allocation_mode=='variable':
                if task.gpu_mode:
                    allocations.append(task.resource_allocations_gpu_mode)
                else:
                    allocations.append(task.resource_allocations_cpu_mode)
            elif self.allocation_mode=='fix':
                allocations.append([task.initial_allocation])
            elif self.allocation_mode=='base':
                pass

        variant_names = list(itertools.product(*variant_names))
        replicas = list(itertools.product(*replicas))
        batches = list(itertools.product(*batches))
        if self.allocation_mode != 'base':
            allocations = list(itertools.product(*allocations))
            all_combinations = list(
                itertools.product(*[
                    variant_names, replicas, batches, allocations]))
        else:
            all_combinations = list(
                itertools.product(*[
                    variant_names, replicas, batches]))

        # generate states header format
        states = self._generate_states()

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
                if self.allocation_mode != 'base':
                    self.pipeline.inference_graph[
                        task_id_i].change_allocation(
                            active_allocation=combination[3][task_id_i])
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
                        self.pipeline.inference_graph[
                            task_id_j].active_variant
                    state[f'task_{task_id_j}_cpu'] =\
                        self.pipeline.inference_graph[task_id_j].cpu
                    state[f'task_{task_id_j}_gpu'] =\
                        self.pipeline.inference_graph[task_id_j].gpu
                    state[f'task_{task_id_j}_cpu_all_replicas'] =\
                        self.pipeline.inference_graph[
                            task_id_j].cpu_all_replicas
                    state[f'task_{task_id_j}_gpu_all_replicas'] =\
                        self.pipeline.inference_graph[
                            task_id_j].gpu_all_replicas
                    state[f'task_{task_id_j}_batch'] =\
                        self.pipeline.inference_graph[task_id_j].batch
                    state[f'task_{task_id_j}_replicas'] =\
                        self.pipeline.inference_graph[task_id_j].replicas
                    state[f'task_{task_id_j}_latency'] =\
                        self.pipeline.inference_graph[task_id_j].latency
                    state[f'task_{task_id_j}_throughput'] =\
                        self.pipeline.inference_graph[task_id_j].throughput
                    state[f'task_{task_id_j}_throughput_all_replicas'] =\
                        self.pipeline.inference_graph[
                            task_id_j].throughput_all_replicas
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
                state['objective'] = self.objective(
                    alpha=alpha, beta=beta, gamma=gamma)
                state['alpha'] = alpha
                state['beta'] = beta
                state['gamma'] = gamma
                states = states.append(state, ignore_index=True)
                if num_state_limit is not None:
                    state_counter += 1
                    if state_counter == num_state_limit: break
        return states

    def brute_force(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None) -> pd.DataFrame:
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha, beta=beta, gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit)
        optimal = states[
            states['objective'] == states['objective'].max()]
        return optimal

    def gurobi_optmizer(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None) -> pd.DataFrame:
        """generate all the possible states based on profiling data

        Args:
            check_constraints (bool, optional): whether to check the
                objective function contraint or not. Defaults to False.
            scaling_cap (int, optional): maximum number of allowed horizontal
                scaling for each node. Defaults to 2.
            alpha (float, optional): accuracy ojbective weight.
                Defaults to 1.
            beta (float, optional): resource usage
                objective weigth. Defaults to 1.
            arrival_rate (int, optional): arrival rate into
                the pipeline. Defaults to None.
            sla (float, optional): end to end service level agreement
                of pipeline. Defaults to None.
            state_limit (int, optional): whether to generate a
                fixed number of state. Defaults to None.

        Returns:
            pd.DataFrame: all the states of the pipeline
        """
        sla = self.pipeline.sla
        if num_state_limit is not None:
            state_counter = 0
        variant_names = []
        replicas = []
        batches = []
        assert self.allocation_mode == 'base',\
            'currrently only base mode is supported with Gurobi'
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            replicas.append(np.arange(1, scaling_cap+1))
            batches.append(task.batches)
        batching_cap = max(batches[0])

        def func_l(batch, params):
            """using parameters of fitted models

            Args:
                batch: batch size
                params: parameters of the linear model

            Returns:
                latency
            """
            latency = params[0] * batch + params[1]
            return latency

        def func_q(batch, params):
            """queueing latency

            Args:
                batch: batch size
                params: parameters of the linear model

            Returns:
                latency
            """
            queue = params[0] * batch
            return queue

        # defining groubipy model for descision problem
        model = gp.Model('pipeline')

        # stages
        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants

        # sets
        gurobi_variants = []
        gurobi_replicas = []
        gurobi_batches = []
        for stage_index, stage_name in enumerate(stages):
            gurobi_variants += [(stage_name, variant) for variant in variant_names[stage_index]]
            gurobi_replicas += [stage_name]
            gurobi_batches += [stage_name]

        # variables
        i = model.addVars(gurobi_variants, name='i', vtype=GRB.BINARY)
        n_lb = 1
        b_lb = 1
        n = model.addVars(gurobi_replicas, name='n', vtype=GRB.INTEGER, lb=n_lb, ub=scaling_cap)
        b = model.addVars(gurobi_batches, name='b', vtype=GRB.INTEGER, lb=b_lb, ub=batching_cap)
        model.update()

        # coefficients
        latency_parameters = self.latency_parameters()
        # throughputs = self.throughputs()
        base_allocations = self.base_allocations()
        queue_parameters = self.queue_parameters()
        accuracy_parameters = self.accuracy_parameters()

        # paper constraints
        model.addQConstr(
            (gp.quicksum(func_l(b[stage], latency_parameters[stage][variant]) * i[stage, variant]\
                for stage in stages for variant in stages_variants[stage]) +\
                    gp.quicksum(func_q(b[stage], queue_parameters[stage]) for stage in stages) <= sla), name='latency')
        # upper bound trick based on
        # https://support.gurobi.com/hc/en-us/community/posts/12996185241105-How-to-add-quadratic-constraint-in-conditional-indicator-constraints
        M = arrival_rate * self.pipeline_latency_upper_bound() - n_lb * b_lb
        for stage in stages:
            for variant in stages_variants[stage]:
                model.addQConstr(((
                    arrival_rate * func_l(b[stage],
                    latency_parameters[stage][variant]) - n[stage] * b[stage]) <= M * (1-i[stage, variant])), f'throughput-{stage}-{variant}')
        model.addConstrs(
            (gp.quicksum(
                i[stage, variant] for variant in stages_variants[stage]) == 1\
                    for stage in stages), name='one_model')

        # objectives
        if self.pipeline.accuracy_method == 'multiply':
            raise NotImplementedError(
                ("multiplication accuracy objective is not implemented",
                 "yet for Grubi due to quadratic limitation of Gurobi"))
        elif self.pipeline.accuracy_method == 'sum':
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][vairant] * i[stage, vairant]\
                    for stage in stages for vairant in stages_variants[stage]
            )
        elif self.pipeline.accuracy_method == 'average':
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][vairant]\
                    * i[stage, vairant] * (1/len(stages))\
                        for stage in stages for vairant in stages_variants[stage]
            )

        resource_objective = gp.quicksum(
            base_allocations[stage][vairant] * n[stage] * i[stage, vairant]\
                for stage in stages for vairant in stages_variants[stage]
        )
        batch_objective = gp.quicksum(
            b[stage] for stage in stages
        )
    
        # update the model
        model.setObjective(
            alpha * accuracy_objective +\
            beta * resource_objective +\
            gamma * batch_objective, GRB.MAXIMIZE)

        # Parameters
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 10**8
        model.Params.PoolGap = 0.0

        model.update()

        # Solve bilinear model
        model.params.NonConvex = 2
        model.optimize()
        # model.display()
        # model.printStatus()
        all_vars = {v.varName: v.x for v in model.getVars()}
        i_var_output = {key: value for key, value in all_vars.items() if 'i[' in key}
        n_var_output = {key: value for key, value in all_vars.items() if 'n[' in key}
        b_var_output = {key: value for key, value in all_vars.items() if 'b[' in key}

        i_output = {} # i_output[stage] <- variant
        for stage in stages:
            i_output[stage] = {}
            for variant in stages_variants[stage]:
                result = [value for key, value in i_var_output.items() if stage in key and variant in key][0]
                if result == 1:
                    i_output[stage] = variant

        n_output = {} # n_output[stage]
        for stage in stages:
            result = [value for key, value in n_var_output.items() if stage in key][0]
            n_output[stage] = result

        b_output = {} # b_output[stage]
        for stage in stages:
            result = [value for key, value in b_var_output.items() if stage in key][0]
            b_output[stage] = result

        # set models, replication and batch of inference graph
        for task_id, stage in enumerate(stages):
            self.pipeline.inference_graph[task_id].model_switch(i_output[stage])
            self.pipeline.inference_graph[task_id].re_scale(n_output[stage])
            self.pipeline.inference_graph[task_id].change_batch(b_output[stage])

        # generate states header format
        states = self._generate_states()

        # generate states data
        state = {}
        for task_id_j in range(self.pipeline.num_nodes):
            # record all stats under this configs
            state[f'task_{task_id_j}_variant'] =\
                self.pipeline.inference_graph[
                    task_id_j].active_variant
            state[f'task_{task_id_j}_cpu'] =\
                self.pipeline.inference_graph[task_id_j].cpu
            state[f'task_{task_id_j}_gpu'] =\
                self.pipeline.inference_graph[task_id_j].gpu
            state[f'task_{task_id_j}_cpu_all_replicas'] =\
                self.pipeline.inference_graph[
                    task_id_j].cpu_all_replicas
            state[f'task_{task_id_j}_gpu_all_replicas'] =\
                self.pipeline.inference_graph[
                    task_id_j].gpu_all_replicas
            state[f'task_{task_id_j}_batch'] =\
                self.pipeline.inference_graph[task_id_j].batch
            state[f'task_{task_id_j}_replicas'] =\
                self.pipeline.inference_graph[task_id_j].replicas
            state[f'task_{task_id_j}_latency'] =\
                self.pipeline.inference_graph[task_id_j].latency
            state[f'task_{task_id_j}_throughput'] =\
                self.pipeline.inference_graph[task_id_j].throughput
            state[f'task_{task_id_j}_throughput_all_replicas'] =\
                self.pipeline.inference_graph[
                    task_id_j].throughput_all_replicas
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
        state['objective'] = self.objective(
            alpha=alpha, beta=beta, gamma=gamma)
        state['alpha'] = alpha
        state['beta'] = beta
        state['gamma'] = gamma
        states = states.append(state, ignore_index=True)
        return states

    def optimize(
        self, optimization_method: str,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int, num_state_limit: int=None):

        if optimization_method == 'brute-force':
            optimal = self.brute_force(
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
        elif optimization_method == 'gurobi':
            optimal = self.gurobi_optmizer(
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit)
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
        return self.pipeline.pipeline_latency < self.pipeline.sla

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