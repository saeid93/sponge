from typing import Dict, List

class ResourceRequirement:
    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        self.cpu = cpu
        self.gpu = gpu

class Profile:
    def __init__(self, batch: int, latency: float) -> None:
        self.batch = batch
        self.latency = latency
    
    @property
    def throughput(self):
        return 1/self.latency

class Model:
    def __init__(
        self,
        name: str,
        resources: ResourceRequirement,
        profiles: List[Profile],
        accuracy: float) -> None:

        self.resources = resources
        self.profiles = profiles
        self.accuracy = accuracy
        self.name = name

class Task:
    def __init__(
        self,
        name: str,
        available_variants: Dict[str, Model],
        active_variant: str, replica: int, batch: int,
        gpu_mode: False) -> None:
        self.available_variants = available_variants
        self.replica = batch
        self.batch = batch
        if active_variant not in self.available_variants.keys():
            raise ValueError(f"{active_variant} not part of variants",
                             f"available variants: {self.available_variants}")
        self.active_variant = active_variant
        self.replica = replica
        self.gpu_mode = gpu_mode
        self.name = name
    # TODO add queue delay

    def model_switch(self, active_variant):
        if active_variant not in self.available_variants.keys():
            raise ValueError(f"{active_variant} not part of variants",
                             f"available variants: {self.available_variants}")
        self.active_variant = active_variant

    def re_scale(self, replica):
        self.replica = replica

    @property
    def active_model(self):
        return self.available_variants[self.active_variant]

    @property
    def cpu(self):
        if self.gpu_mode:
            raise ValueError('The node is on gpu mode')
        else:
            return self.active_model.resources.cpu * self.replica

    @property
    def gpu(self):
        if self.gpu_mode:
            return self.active_model.resources.gpu * self.replica

    @property
    def latency(self):
        # TODO include queuing latency
        return self.active_model.profiles[self.batch].latency

    @property
    def throughput(self):
        return self.active_model.profiles[self.batch].throughput\
            * self.replica

    @property
    def accuracy(self):
        return self.active_model.accuracy

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
            map(lambda l: l.throughput, self.inference_graph))
        return throughputs

    @property
    def stage_wise_latencies(self):
        latencies = list(
            map(lambda l: l.latency, self.inference_graph))
        return latencies

    @property
    def stage_wise_replicas(self):
        replicas = list(
            map(lambda l: l.replica, self.inference_graph))
        return replicas

    @property
    def stage_wise_cpu_usage(self):
        cpu = []
        for task in self.inference_graph:
            if not task.gpu_mode:
                cpu.append(task.cpu)
            else:
                cpu.append(0)
        return cpu

    @property
    def stage_wise_gpu_usage(self):
        gpu = []
        for task in self.inference_graph:
            if task.gpu_mode:
                gpu.append(task.gpu)
            else:
                gpu.append(0)
        return gpu

    @property    
    def pipeline_latency(self):
        return sum(self.stage_wise_latencies)

    @property    
    def accuracy(self):
        accuracy = 1
        for task in self.inference_graph:
            accuracy *= task.accuracy
        return accuracy

    @property
    def cpu_usage(self):
        return sum(self.stage_wise_cpu_usage)

    @property
    def gpu_usage(self):
        return sum(self.stage_wise_gpu_usage)

    def visualize(self):
        pass

class Optimizer:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def all_states(self):
        """
        returns all states of the pipeline
        with 
        """
        pass

    def all_feasible_states(self):
        pass

    def greedy_optimizer(self):
        pass

    def objective(
        self,
        alpha: float, beta: float) -> None:
        objective = alpha * self.pipeline.accuracy + beta * 1/self.pipeline.cpu_usage # TODO make it pipeline specification
        return objective

    def can_sustain_load(
        self,
        arrival_rate: int) -> bool:
        """
        whether the existing config can sustain a load
        """
        for task in self.pipeline.inference_graph:
            if arrival_rate > task.throughput:
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
            raise ValueError(f'The load can be sustained!')
        bottlenecks = []
        for task_id, task in enumerate(self.pipeline.inference_graph):
            if arrival_rate > task.throughput:
                bottlenecks.append(task_id)
        return bottlenecks

    def sla_is_met(self) -> bool:
        return self.pipeline.pipeline_latency < self.pipeline.sla
