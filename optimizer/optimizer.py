import os
import random
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import math
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import itertools
from copy import deepcopy
from .models import Pipeline


class Optimizer:
    def __init__(
        self,
        pipeline: Pipeline,
        allocation_mode: str,
        complete_profile: bool,
        only_measured_profiles: bool,
        random_sample: bool,
        baseline_mode: Optional[str] = None,
    ) -> None:
        """_summary_

        Args:
            pipeline (Pipeline): pipeline objecit for optimization
            allocation_mode (str): allocation mode for cpu usage,
                fix | base | variable
                fix: stays on the initiial CPU allocation
                base: finding the base allocation explained in the paper
                variable: search through the cpu allocation as a configuration knob
            complete_profile (bool): whether to log the complete result or not
            only_measured_profiles (bool): only profiled based on the measured latency/throughput
                profiles and not using regression models
        """
        self.pipeline = pipeline
        self.allocation_mode = allocation_mode
        self.complete_profile = complete_profile
        self.only_measured_profiles = only_measured_profiles
        self.random_sample = random_sample
        self.baseline_mode = baseline_mode

        # TODO initialize latency model
        # TODO extract data from the all_states
        states = self.all_states(
            check_constraints=False, # Getting all possible values
            scaling_cap=10000, # Dummy number
            alpha=10000, # Dummy number
            arrival_rate=1000, # Dummy number
            num_state_limit=1000, # Dummy number
            only_vertical=True,
            batching_cap=1000, # Dummy number
        )
        b_data = states['task_0_batch'].tolist()
        c_data = states['task_0_cpu'].tolist()
        l_data = [int(i * 1000) for i in states['task_0_latency'].tolist()] # all latency in milicores
        params, _ = curve_fit(self.batch_cost_latency_model, (b_data, c_data), l_data)
        self.gamma, self.delta, self.epsilon, self.eta = params  # eq 2 parameters

    def resource_objective(self) -> float:
        """
        objective function of the pipeline
        """
        resource_objective = self.pipeline.cpu_usage
        return resource_objective

    def batch_cost_latency_model(self, bc, gamma, delta, epsilon, eta):
        b, c = bc
        return gamma * b / c + delta * b + epsilon / c + eta

    def batch_cost_latency_calculation(self, b, c, gamma, delta, epsilon, eta):
        return int(gamma * b / c + delta * b + epsilon / c + eta)

    def batch_objective(self) -> float:
        """
        batch objecive of the pipeline
        """
        max_batch = 0
        for task in self.pipeline.inference_graph:
            max_batch += task.batch
        return max_batch

    def objective(self, alpha: float) -> Dict[str, float]:
        """
        objective function of the pipeline
        """
        objectives = {}
        objectives["resource_objective"] = self.resource_objective()
        objectives["batch_objective"] = alpha * self.batch_objective()
        objectives["objective"] = (
            objectives["resource_objective"]
            + objectives["batch_objective"]
        )
        return objectives

    def constraints(self, arrival_rate: int) -> bool: # TODO add the new sla
        """
        whether the constraints are met or not
        """
        if self.sla_is_met() and self.can_sustain_load(arrival_rate=arrival_rate):
            return True
        return False

    def all_states(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        check_constraints: bool,
        arrival_rate: int,
        only_vertical: bool,
        num_state_limit: int = None,
        complete_profile: bool = True
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
        if num_state_limit is not None:
            state_counter = 0
        batches = []
        allocations = []
        replicas = []
        for task in self.pipeline.inference_graph:
            batches.append([batch for batch in task.batches if batch < batching_cap])
            allocations.append(task.resource_allocations_cpu_mode)
            if only_vertical:
                replicas.append(np.array([1]))
            else:
                replicas.append(np.arange(1, scaling_cap + 1))

        batches = list(itertools.product(*batches))
        allocations = list(itertools.product(*allocations))
        replicas = list(itertools.product(*replicas))
        all_combinations = itertools.product(*[allocations, batches, replicas])

        if self.random_sample:
            all_combinations = random.sample(list(all_combinations), num_state_limit)

        # generate states header format
        states = []

        for combination in all_combinations:
            try:  # Not all models profiles are available under all batch sizes
                for task_id_i in range(self.pipeline.num_nodes):
                    self.pipeline.inference_graph[task_id_i].change_allocation(
                        active_allocation=combination[0][task_id_i]
                    )
                    self.pipeline.inference_graph[task_id_i].change_batch(
                        batch=combination[1][task_id_i]
                    )
                    self.pipeline.inference_graph[task_id_i].re_scale(
                        replica=combination[2][task_id_i]
                    )
                ok_to_add = False
                if check_constraints:
                    if self.constraints(arrival_rate=arrival_rate): # TODO add slas
                        ok_to_add = True
                else:
                    ok_to_add = True
                if ok_to_add:
                    state = {}
                    if complete_profile:
                        for task_id_j in range(self.pipeline.num_nodes):
                            # record all stats under this configs
                            state[
                                f"task_{task_id_j}_latency"
                            ] = self.pipeline.inference_graph[task_id_j].latency
                            state[
                                f"task_{task_id_j}_throughput"
                            ] = self.pipeline.inference_graph[task_id_j].throughput
                            state[
                                f"task_{task_id_j}_throughput_all_replicas"
                            ] = self.pipeline.inference_graph[
                                task_id_j
                            ].throughput_all_replicas
                            state[
                                f"task_{task_id_j}_measured"
                            ] = self.pipeline.inference_graph[task_id_j].measured
                        state["pipeline_latency"] = self.pipeline.pipeline_latency
                        state["pipeline_throughput"] = self.pipeline.pipeline_throughput
                        state["pipeline_cpu"] = self.pipeline.pipeline_cpu
                        state["alpha"] = alpha
                        state["resource_objective"] = self.resource_objective()
                        state["batch_objective"] = self.batch_objective()

                    for task_id_j in range(self.pipeline.num_nodes):
                        # record all stats under this configs
                        state[
                            f"task_{task_id_j}_variant"
                        ] = self.pipeline.inference_graph[task_id_j].active_variant
                        state[f"task_{task_id_j}_cpu"] = self.pipeline.inference_graph[
                            task_id_j
                        ].cpu
                        state[
                            f"task_{task_id_j}_batch"
                        ] = self.pipeline.inference_graph[task_id_j].batch
                        state[
                            f"task_{task_id_j}_replicas"
                        ] = self.pipeline.inference_graph[task_id_j].replicas

                    state["objectives"] = self.objective(
                        alpha=alpha,
                    )
                    state["objective"] = self.objective(
                        alpha=alpha,
                    )['objective']
                    states.append(state)
                    if num_state_limit is not None:
                        state_counter += 1
                        # print(f"state {state_counter} added")
                        if state_counter == num_state_limit:
                            break
            except StopIteration:
                pass
        return pd.DataFrame(states)

    def dynainf(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        arrival_rate: int,
        sla_series: List[float],
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
            only_vertical=True,
            batching_cap=batching_cap
        )
        optimal = states[states["objective"] == states["objective"].max()]
        return optimal

    def fa2(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        arrival_rate: int,
        sla_series: List[float],
        num_state_limit: int,
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
            arrival_rate (int, optional): arrival rate into
                the pipeline. Defaults to None.
            sla (float, optional): end to end service level agreement
                of pipeline. Defaults to None.
            baseline: baseline approach [scaling | switch]
        Returns:
            pd.DataFrame: all the states of the pipeline
        """

        # states = self.all_states(
        #     check_constraints=True,
        #     scaling_cap=scaling_cap,
        #     alpha=alpha,
        #     arrival_rate=arrival_rate,
        #     num_state_limit=num_state_limit,
        #     only_vertical=False,
        #     batching_cap=batching_cap
        # )
        # optimal = states[states["objective"] == states["objective"].max()]
        b_max = batching_cap  # max batch size configuration
        RPS = arrival_rate  # workload
        # q = [50] * RPS  # calculate this from the user
        q = sla_series[-1]
        # SLO = 1000  # default SLO
        SLO = self.pipeline.sla  # default SLO
        # cl_max = max(q)  # maximum communication latency
        cl_max = SLO - q
        instance_number = 100  # result number of instances
        best_batch = 0  # result batch size
        SECOND_MILISECOND = 1000
        for b in range(1, b_max + 1):  # iterate over all the batch sizes
            l_bc = self.batch_cost_latency_calculation(b, 1, self.gamma, self.delta, self.epsilon, self.eta)  # calculate latency with the candidate batch and cpu using eq 2
            q_time = 0  # queue time for requests
            if l_bc > SECOND_MILISECOND:
                break
            curr_instance = math.ceil(RPS / (int(SECOND_MILISECOND / l_bc) * b))  # current instance nubmer
            for i in range(0, RPS, b):  # iterate over all the requests in the queue
                if l_bc + q_time + cl_max < SLO and curr_instance < instance_number:  # the current configuration not satisfy the SLOs and there is a smaller instance number
                    instance_number = curr_instance
                    best_batch = b
                q_time += l_bc  # increase queuing time for the next batch of request

        optimal_dict = {'task_0_cpu': [1], 'task_0_replicas': [instance_number], 'task_0_batch': [best_batch], 'objective': [0]}
        optimal = pd.DataFrame(optimal_dict) 
        return optimal
        # return optimal


    def optimize(
        self,
        optimization_method: str,
        scaling_cap: int,
        cpu_cap: int,
        alpha: float,
        arrival_rate: int,
        sla_series: List[float],
        num_state_limit: int = None,
        batching_cap: int = None
    ) -> pd.DataFrame:
        if optimization_method == "dynainf":
            optimal = self.dynainf(
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                arrival_rate=arrival_rate,
                sla_series=sla_series,
                num_state_limit=num_state_limit,
            )
        elif optimization_method == "fa2":
            optimal = self.fa2(
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                arrival_rate=arrival_rate,
                sla_series=sla_series,
                num_state_limit=num_state_limit,
            )
        else:
            raise ValueError(f"Invalid optimization_method: {optimization_method}")
        return optimal

    def can_sustain_load(self, arrival_rate: int) -> bool:
        """
        whether the existing config can sustain a load
        """
        for task in self.pipeline.inference_graph:
            if arrival_rate > task.throughput_all_replicas:
                return False
        return True

    def sla_is_met(self) -> bool:
        return self.pipeline.pipeline_latency < self.pipeline.sla

