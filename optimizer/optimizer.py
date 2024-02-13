import os
import random
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
        sla_series: List[float],
        only_vertical: bool,
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
                    if self.complete_profile:
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
            sla_series=sla_series,
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
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            arrival_rate=arrival_rate,
            sla_series=sla_series,
            num_state_limit=num_state_limit,
            batching_cap=batching_cap
        )
        optimal = states[states["objective"] == states["objective"].max()]
        return optimal


    def optimize(
        self,
        optimization_method: str,
        scaling_cap: int,
        alpha: float,
        arrival_rate: int,
        sla_series: List[float],
        num_state_limit: int = None,
        batching_cap: int = None,
        dir_path: str = None,
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
                dir_path=dir_path,
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

