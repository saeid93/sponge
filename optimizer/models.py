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
        self.normalized_accuracy = None
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
        poly_features = PolynomialFeatures(degree=2)
        train_x_poly = poly_features.fit_transform(train_x)
        test_x_poly = poly_features.transform(test_x)

        latency_model = LinearRegression()
        latency_model.fit(train_x_poly, train_y)

        test_y = latency_model.predict(test_x_poly)

        predicted_profiles = []
        for index, x, y in zip(
            range(len(all_x)), test_x.reshape(-1), test_y.reshape(-1)
        ):
            if self.only_measured_profiles:
                predicted_profiles.append(
                    Profile(
                        batch=x,
                        latency=y,
                        measured=True,
                        measured_throughput=self.measured_profiles[
                            index
                        ].measured_throughput,
                    )
                )
            else:
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

        model_parameters = {
            "coefficients": coefficients,
            "intercept": intercept,
            # "x_poly": test_x_poly,
        }

        # HACK only power of twos for now
        selected_profiles_indices = [2**i - 1 for i in range(int(math.log2(len(profiles))) + 1)]
        profiles = [profiles[index] for index in selected_profiles_indices if index < len(profiles)]

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
        gpu_mode: False,
    ) -> None:
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
        self.variants_accuracies_normalized = self.find_variants_accuracies_normalized()
        if allocation_mode == "base":
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
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

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
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f"of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}"
            )

        if self.allocation_mode == "base":
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
        # TOOD change here
        # 1. filter out models
        for model_variant in self.variant_names:
            for allocation in self.available_model_profiles:
                if allocation.name == model_variant:
                    models[model_variant].append(allocation)
        base_allocation = {}
        for model_variant, allocations in models.items():
            # finding the minimum allocation that can respond
            # to the threshold
            # the profiles are sorted therefore therefore
            # we iterate from the first profile
            for allocation in allocations:
                # check if the max batch size throughput
                # can reponsd to the threshold
                if (
                    allocation.profiles[-1].throughput >= self.threshold
                    and allocation.profiles[-1].throughput >= self.sla
                ):
                    base_allocation[model_variant] = deepcopy(
                        allocation.resource_allocation
                    )
                    break
            else:  # no-break
                # TODO remove none-working models
                raise ValueError(
                    f"No responsive model profile to threshold {self.threshold}"
                    f" or model sla {self.sla} was found"
                    f" for model variant {model_variant} "
                    "consider either changing the the threshold or "
                    f"sla factor {self.sla_factor}"
                )
        return base_allocation

    def set_to_base_allocation(self):
        self.change_allocation(
            active_allocation=self.base_allocations[self.active_variant]
        )

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
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {self.active_variant} and allocation"
                f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

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
        accuracies_normalized = (
            np.arange(len(accuracies)) / (len(accuracies) - 1)
        ).tolist()
        variants_accuracies_normalized = {
            variant: accuracy_normalized
            for variant, accuracy_normalized in zip(variants, accuracies_normalized)
        }
        return variants_accuracies_normalized

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
        if self.gpu_mode:
            raise ValueError("The node is on gpu mode")
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
            raise ValueError("The node is on gpu mode")
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
        latency = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).latency
        return latency

    @property
    def latency(self) -> float:
        latency = self.model_latency + self.queue_latency
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
    def accuracy(self):
        if self.normalize_accuracy:
            return self.variants_accuracies_normalized[self.active_variant]
        else:
            return self.active_model.accuracy

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
                        f"pipeline is deployed on cpu",
                        f"but task {task.name} is on gpu",
                    )

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
    def sla(self):
        sla = sum(map(lambda l: l.sla, self.inference_graph))
        return sla

    @property
    def stage_wise_accuracies(self):
        latencies = list(map(lambda l: l.accuracy, self.inference_graph))
        return latencies

    @property
    def stage_wise_replicas(self):
        replicas = list(map(lambda l: l.replicas, self.inference_graph))
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
                accuracy = task.variants_accuracies_normalized[acive_variant]
            else:
                accuracy = task.variants_accuracies[acive_variant]
            tasks_accuracies[acive_variant] = accuracy
        if self.accuracy_method == "multiply":
            accuracy = 1
            for task, task_accuracy in tasks_accuracies.items():
                accuracy *= task_accuracy
        elif self.accuracy_method == "sum":
            accuracy = 0
            for task, task_accuracy in tasks_accuracies.items():
                accuracy += task_accuracy
        elif self.accuracy_method == "average":
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
