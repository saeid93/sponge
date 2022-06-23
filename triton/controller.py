"""
a minimal wrapper that laod/unload/switch
betwee models in Nvidia Triton Server
inspired by
https://github.com/saeid93/mobile-kube/tree/main/mobile-kube/src/mobile_kube/util/kubernetes_utils
"""

import requests
import networkx as nx
# import seldon_core <if using seldon>
# from kubernetes.client import V1Node, V1Pod, V1Service <if k8s>

class Cluster:
    # k8s cluster
    def __init__(self) -> None:
        self.nodes = [] # list of cluster nodes
        self.seldon_graphs = [] # list of Seldon graphs in the cluster


class Nodes:
    # k8s cluster nodes
    def __init__(self) -> None:
        self.triton_nodes: TritonNodes = []


class SeldonGraph:
    def __init__(self) -> None:
        self.nodes: TritonNodes = []
        self.graph = nx.Graph() # Seldon core nodes dependancy graph
    
    def list_nodes(self):
        # list all nodes in the Seldon graph
        pass

    def list_nodes(self):
        pass

    @property
    def graph_dependancy(self):
        pass

class Metrics:
    # connect to the prometheus for metrics
    def __init__(self) -> None:
        pass

    def metrics(self):
        pass


class TritonNodes:
    def __init__(self) -> None:
        self.models = []
        self.graph = nx.Graph()
        pass

    def load_model(model_name: str):
        pass

    def unload_model(model_name: str):
        pass

    def list_nodes(model_name: str):
        # list all nodes in a single Triton node
        pass

    @property
    def graph_dependancy(self):
        pass


