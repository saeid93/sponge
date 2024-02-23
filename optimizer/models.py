from typing import Dict, List, Union
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import math


class ResourceAllocation:
    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        # For now only one type CPU/GPU allocation is allowed
        if cpu != 0 and gpu != 0:
            raise ValueError("For now only one of the CPU or GPU allocation is allowed")
        self.cpu = cpu
        self.gpu = gpu


class Profile:
    def __init__(
        self,
        batch: int,
        latency: float,
        measured: bool = True,
        measured_throughput=None,
    ) -> None:
        self.batch = batch
        self.latency = latency
        self.measured = measured
        if measured_throughput is not None:
            self.measured_throughput = measured_throughput

    @property
    def throughput(self):
        if self.measured:
            throughput = self.measured_throughput
        else:
            throughput = (1 / self.latency) * self.batch
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
        only_measured_profiles: bool,
        accuracy: float,
    ) -> None:
        self.resource_allocation = resource_allocation
        self.measured_profiles = measured_profiles
        self.measured_profiles.sort(key=lambda profile: profile.batch)
        self.accuracy = accuracy / 100
        self.name = name
        self.only_measured_profiles = only_measured_profiles
        self.profiles, self.latency_model_params = self.regression_model()

    def regression_model(self) -> Union[List[Profile], Dict[str, float]]:
        """
        interapolate the latency for unknown batch sizes
        """
        train_x = np.array(
            list(map(lambda l: l.batch, self.measured_profiles))
        ).reshape(-1, 1)
        train_y = np.array(
            list(map(lambda l: l.latency, self.measured_profiles))
        ).reshape(-1, 1)
        if self.only_measured_profiles:
            all_x = train_x
        else:
            all_x = np.arange(self.min_batch, self.max_batch + 1)
        # HACK all the data from the latency model and not using
        # measured data
        # test_x = all_x[~np.isin(all_x, train_x)].reshape(-1, 1)
        test_x = all_x.reshape(-1, 1)
        profiles = []
        if self.only_measured_profiles:
            for index, x, y in zip(
                range(len(all_x)), train_x.reshape(-1), train_y.reshape(-1)
            ):
                profiles.append(
                    Profile(
                        batch=x,
                        latency=self.measured_profiles[index].latency,
                        measured=True,
                        measured_throughput=self.measured_profiles[
                            index
                        ].measured_throughput,
                    )
                )
            model_parameters = {"coefficients": None, "intercept": None}
        else:
            poly_features = PolynomialFeatures(degree=2)
            train_x_poly = poly_features.fit_transform(train_x)
            test_x_poly = poly_features.transform(test_x)

            latency_model = LinearRegression()
            latency_model.fit(train_x_poly, train_y)

            test_y = latency_model.predict(test_x_poly)

            # TODO add a hueristic to remove the <0 latency values
            # we set polynomial as reference but for small values
            # polynomial will result into negative values
            # if there is a negative values in the polynomial results
            # we fill it with linear model resutls
            # test_x = all_x.reshape(-1, 1)
            latency_model_linear = LinearRegression()
            latency_model_linear.fit(train_x, train_y)
            test_y_linear = latency_model_linear.predict(test_x)

            for index, lateny in enumerate(test_y):
                if lateny < 0:
                    test_y[index] = test_y_linear[index]

            predicted_profiles = []
            for index, x, y in zip(
                range(len(all_x)), test_x.reshape(-1), test_y.reshape(-1)
            ):
                predicted_profiles.append(
                    Profile(
                        batch=x, latency=y, measured=False, measured_throughput=None
                    )
                )
            profiles: List[Profile] = predicted_profiles
            profiles.sort(key=lambda profile: profile.batch)

            # Extract coefficients and intercept
            coefficients = latency_model.coef_[0]
            intercept = latency_model.intercept_

            model_parameters = {"coefficients": coefficients, "intercept": intercept}

            # HACK only power of twos for now
            # if not self.only_measured_profiles:
            selected_profiles_indices = [
                2**i - 1 for i in range(int(math.log2(len(profiles))) + 1)
            ]
            profiles = [
                profiles[index]
                for index in selected_profiles_indices
                if index < len(profiles)
            ]

        return profiles, model_parameters

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
        replica: int,
        batch: int,
        allocation_mode: str,
        threshold: int,
        sla_factor: int,
        normalize_accuracy: bool,
        lowest_model_accuracy: float = 0,
    ) -> None:
        self.available_model_profiles = available_model_profiles
        self.active_variant = active_variant
        self.active_allocation = active_allocation
        self.initial_allocation = active_allocation
        self.replicas = replica
        self.batch = batch
        self.replicas = replica
        self.normalize_accuracy = normalize_accuracy
        self.threshold = threshold
        self.name = name
        self.sla_factor = sla_factor
        self.allocation_mode = allocation_mode
        self.lowest_model_accuracy = lowest_model_accuracy

        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.active_allocation.cpu == variant.resource_allocation.cpu:
                    self.active_variant_index = variant_index
                    break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f" of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

    def remove_model_profiles_by_name(self, model_name: str):
        self.available_model_profiles = [
            profile
            for profile in self.available_model_profiles
            if profile.name != model_name
        ]

    def get_all_models_by_name(self, model_name: str):
        return [
            profile
            for profile in self.available_model_profiles
            if profile.name == model_name
        ]

    def add_model_profile(self, model: Model):
        self.available_model_profiles.append(model)

    def add_model_profiles(self, model: List[Model]):
        self.available_model_profiles += model

    def model_switch(self, active_variant: str) -> None:
        """
        changes variant under specific allocation
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.active_allocation.cpu == variant.resource_allocation.cpu:
                    self.active_variant_index = variant_index
                    self.active_variant = active_variant
                    break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f"of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}"
            )

    @property
    def num_variants(self):
        return len(self.variant_names)

    def change_allocation(self, active_allocation: ResourceAllocation) -> None:
        """
        change allocation of a specific variant
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == self.active_variant:
                if active_allocation.cpu == variant.resource_allocation.cpu:
                    self.active_variant_index = variant_index
                    self.active_allocation = active_allocation
                    break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {self.active_variant} and allocation"
                f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

    def re_scale(self, replica) -> None:
        self.replicas = replica

    def change_batch(self, batch) -> None:
        self.batch = batch

    @property
    def active_model(self) -> Model:
        return self.available_model_profiles[self.active_variant_index]

    @property
    def latency_model_params(self) -> Model:
        return self.available_model_profiles[
            self.active_variant_index
        ].latency_model_params

    @property
    def cpu(self) -> int:
        return self.active_model.resource_allocation.cpu

    @property
    def cpu_all_replicas(self) -> int:
        return self.active_model.resource_allocation.cpu * self.replicas

    @property
    def queue_latency(self) -> float:
        # TODO TEMP
        queue_latency = 0
        return queue_latency

    @property
    def model_latency(self) -> float:
        latency = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).latency
        return latency

    @property
    def latency(self) -> float:
        latency = self.model_latency # + self.queue_latency
        return latency

    @property
    def throughput(self) -> float:
        throughput = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).throughput
        return throughput

    @property
    def measured(self) -> bool:
        measured = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).measured
        return measured

    @property
    def throughput_all_replicas(self):
        return self.throughput * self.replicas

    @property
    def variant_names(self):
        return list(set(map(lambda l: l.name, self.available_model_profiles)))

    @property
    def batches(self):
        batches = list(map(lambda l: l.batch, self.active_model.profiles))
        return batches

    @property
    def resource_allocations_cpu_mode(self):
        cpu_allocations = list(
            set(
                list(
                    map(
                        lambda l: l.resource_allocation.cpu,
                        self.available_model_profiles,
                    )
                )
            )
        )
        resource_allocations = list(
            map(lambda l: ResourceAllocation(cpu=l), cpu_allocations)
        )
        return resource_allocations

    @property
    def resource_allocations_gpu_mode(self):
        gpu_allocations = list(
            set(
                list(
                    map(
                        lambda l: l.resource_allocation.gpu,
                        self.available_model_profiles,
                    )
                )
            )
        )
        resource_allocations = list(
            map(lambda l: ResourceAllocation(gpu=l), gpu_allocations)
        )
        return resource_allocations


class Pipeline:
    def __init__(
        self,
        inference_graph: List[Task],
        gpu_mode: bool,
        sla_factor: int,
        accuracy_method: str,
        normalize_accuracy: bool,
        sla: int,
    ) -> None:
        self.inference_graph: List[Task] = inference_graph
        self.gpu_mode = gpu_mode
        self.sla_factor = sla_factor
        self.accuracy_method = accuracy_method
        self.normalize_accuracy = normalize_accuracy
        self.sla = sla

    def add_task(self, task: Task):
        self.inference_graph.append(task)

    def remove_task(self):
        self.inference_graph.pop()

    @property
    def stage_wise_throughput(self):
        throughputs = list(
            map(lambda l: l.throughput_all_replicas, self.inference_graph)
        )
        return throughputs

    @property
    def stage_wise_latencies(self):
        latencies = list(map(lambda l: l.latency, self.inference_graph))
        return latencies

    @property
    def stage_wise_slas(self):
        slas = dict(map(lambda l: (l.name, l.sla), self.inference_graph))
        return slas

    @property
    def stage_wise_replicas(self):
        replicas = list(map(lambda l: l.replicas, self.inference_graph))
        return replicas

    @property
    def stage_wise_cpu(self):
        cpu = []
        for task in self.inference_graph:
            cpu.append(task.cpu_all_replicas)
        return cpu

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
    def pipeline_latency(self):
        return sum(self.stage_wise_latencies)

    @property
    def pipeline_throughput(self):
        return min(self.stage_wise_throughput)

    @property
    def cpu_usage(self):
        return sum(self.stage_wise_cpu)

    @property
    def num_nodes(self):
        return len(self.inference_graph)

    def visualize(self):
        pass
