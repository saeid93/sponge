# Copyright 2020 Kamran Razavi and Lin Wang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gurobipy as gp
from gurobipy import GRB
from My_Optimizer import MyOptimizer
import copy
import math


class Path:
    def __init__(self, path, sla, workload):
        self.path = path
        self.sla = sla
        self.workload = workload


class GurobiOptimizer:

    @staticmethod
    def func_d(b, para):
        return para[0] * b * b + para[1] * b + para[2]
        # return para[1] * b + para[2]

    @staticmethod
    def func_h(b, para):
        return 1000 * b / (para[0] * b * b + para[1] * b + para[2])

    def __init__(self, wl):
        self.paths = copy.deepcopy(MyOptimizer.default_paths)
        self.model_port = copy.deepcopy(MyOptimizer.default_ports)
        self.ms_limits = []
        self.ms_workload = []
        self.vertex_number = 10
        self.final_bs = []
        self.final_ns = []
        self.paras = MyOptimizer.parameters
        for i in range(len(self.paths)):
            pa = []
            for j in self.paths[i].path:
                pa.append(self.model_port[j])
            if tuple(pa) in wl:
        #        print(tuple(pa))
                self.paths[i].workload += wl[tuple(pa)]
        #for p in self.paths:
        #    print(p.path, p.sla, p.workload)
        self.find_ms_properties()

    def find_ms_properties(self):
        while len(self.ms_limits) > 0:
            del self.ms_limits[0]
            del self.ms_workload[0]
        for i in range(self.vertex_number):
            lim = 10000
            wl = 0
            for p in self.paths:
                if i in p.path:
                    lim = min(lim, p.sla)
                    wl += p.workload

            self.ms_limits.append(lim)
            self.ms_workload.append(wl)
        #print('MS SLA:', self.ms_limits)
        #print('MS Workload:', self.ms_workload, '\n')

    def run(self):
        num_ms = self.vertex_number
        # delta = 0.01
        gp.setParam("LogToConsole", 0)
        gp.setParam('OutputFlag', 0)
        gp.setParam('Threads', 1)
        # gp.setParam('MIPFocus', 3)
        # gp.setParam('PoolSearchMode', 2)
        # gp.setParam('PoolSolutions', 20)
        # gp.setParam('Presolve', 0)
        # create a new model
        model = gp.Model("Inference")

        # create variables
        # n: number of instances per microservice
        # b: batch size for each microservice
        # d: latency for each microservice
        n = model.addVars(num_ms, vtype=GRB.INTEGER, name="n")
        b = model.addVars(num_ms, vtype=GRB.INTEGER, name="b")
        # d = model.addVars(num_ms, name="d")

        # model.addQConstrs(d[i] == self.func_d(b[i], self.paras[i]) for i in range(num_ms) if self.ms_workload[i] > 0)
        # for i in range(num_ms):
        #     model.addQConstr(d[i] == self.func_d(b[i], self.paras[i]))
        # set objective:
        model.setObjective(sum(n[i] for i in range(num_ms)), GRB.MINIMIZE)
        b_max = 16
        model.addConstrs(b[i] <= b_max for i in range(num_ms) if self.ms_workload[i] > 0)
        # model.addConstr(b[0] - 4 == 0)
        # model.addConstr(b[5] - 8 == 0)
        # sla
        for i in range(len(self.paths)):
            latencies = 0
            queues = 0
            for j in range(len(self.paths[i].path)):
                latencies += self.func_d(b[self.paths[i].path[j]], self.paras[self.paths[i].path[j]])
                queues += (1000 * (b[self.paths[i].path[j]] - 1.0)) / self.ms_workload[self.paths[i].path[j]]
                if j < len(self.paths[i].path) - 1:
                    cq = (1000 * (b[self.paths[i].path[j]] - 1.0)) / self.ms_workload[self.paths[i].path[j]]
                    nq = (1000 * (b[self.paths[i].path[j + 1]] - 1.0)) / self.ms_workload[self.paths[i].path[j + 1]]
                    model.addConstr(cq - nq <= 0)
            model.addQConstr(latencies + queues <= self.paths[i].sla)

        # throughput
        for i in range(self.vertex_number):
            if self.ms_workload[i] > 0:
                model.addQConstr((1000 * b[i] * n[i]) - (self.ms_workload[i] * self.func_d(b[i], self.paras[i])) >= 0)

        # Solve bilinear model
        model.params.NonConvex = 2
        model.optimize()
        # model.display()
        # model.printStats()
        # model.printQuality()
        for v in model.getVars():
            if 'n' in v.varName:
                self.final_ns.append(int(v.x))
            elif 'b' in v.varName:
                if int(v.x) == 0:
                    self.final_bs.append(1)
                else:    
                    self.final_bs.append(int(v.x))
            
        total_ns = 0
        latency = []
        throughput = []
        for i in range(0, len(self.final_ns)):
            latency.append(self.func_d(self.final_bs[i], self.paras[i]))
            throughput.append(self.func_h(self.final_bs[i], self.paras[i]) * self.final_ns[i])
            total_ns += self.final_ns[i]
        # print('*' * 3, 'Gurobi', '*' * 3)
        # print('Final_BS:', self.final_bs)
        # print('Final_NS:', self.final_ns)
        # print('Total Instances:', total_ns)
        # self.beautiful_print()

    def beautiful_print(self):
        print('*' * 10, 'RESULT', '*' * 10, '\n')
        print('Path timing:')
        for p in self.paths:
            timing = 0
            for cp in p.path:
                timing += math.ceil(self.func_d(self.final_bs[cp], self.paras[cp]))
                timing += (1000 * (self.final_bs[cp] - 1.0)) / self.ms_workload[cp]
            print(p.path, p.workload, p.sla, timing)
        print()
        print('name b_s n_s latency throughput workload queue')
        ms_names = ['objd', 'objr', 'facr', 'quan', 'imgs', 'alpr', 'tcls', 'nsfw', 'autt', 'summ']
        for v in range(self.vertex_number):
            pr = self.func_d(self.final_bs[v], self.paras[v])
            throughput = self.final_bs[v] * (1000 / pr)
            queue_waiting = 0
            if self.ms_workload[v] > 0:
                queue_waiting = int((1000.0 * (self.final_bs[v] - 1)) / self.ms_workload[v])
            print('{0:4s}{1:4d}{2:4d}{3:8.2f}{4:8.2f}{5:9d}{6:7d}'.format(ms_names[v], self.final_bs[v],
                                                                          self.final_ns[v], pr, self.final_ns[v] * throughput, self.ms_workload[v], queue_waiting))
        print()
        print('Final_BS:', self.final_bs)
        print('Final_NS:', self.final_ns)

        print('Total Instances:', sum(self.final_ns))
