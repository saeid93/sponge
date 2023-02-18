from typing import Dict, List
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import itertools
from copy import deepcopy
from models import (
    Pipeline
)

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
            beta * self.resource_objective() -\
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
            else:
                raise ValueError(
                    f'Invalid allocation_mode: {self.allocation_mode}')

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
                state['batch_objective'] =\
                    self.batch_objective()
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
        else:
            raise ValueError(f'Invalid accuracy method {self.pipeline.accuracy_method}')

        resource_objective = gp.quicksum(
            base_allocations[stage][vairant] * n[stage] * i[stage, vairant]\
                for stage in stages for vairant in stages_variants[stage]
        )
        batch_objective = gp.quicksum(
            b[stage] for stage in stages
        )
    
        # update the model
        model.setObjective(
            alpha * accuracy_objective -\
            beta * resource_objective -\
            gamma * batch_objective, GRB.MAXIMIZE)

        # Parameters for retrieving more than one solution
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 10**8
        model.Params.PoolGap = 0.0

        model.update()

        # Solve bilinear model
        model.params.NonConvex = 2
        model.optimize()
        # model.display()
        # model.printStatus()

        # generate states header format
        states = self._generate_states()

        for solution_count in range(model.SolCount):
            model.Params.SolutionNumber = solution_count
            print([var.Xn for var in model.getVars()])
            print({v.varName: v.Xn for v in model.getVars()})
            all_vars = {v.varName: v.Xn for v in model.getVars()}
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
            state['resource_objective'] =\
                self.resource_objective()
            state['batch_objective'] =\
                self.batch_objective()
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
        else:
            raise ValueError(
                f'Invalid optimization_method: {optimization_method}')
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