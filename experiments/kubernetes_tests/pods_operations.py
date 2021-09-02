"""scripts to check the functionalities of the environment
   for pod operations
"""

from smart_kube import Cluster
from time import sleep
from smart_kube.cluster import clean_all_namespaces


clean_all_namespaces()
# --------------- cluster operations ---------------
cluster = Cluster(namespace='pods-operations')
print("\n")

# --------------- pods operations ---------------
cluster.create_experiment_pod(
    name='experimental-pod',
    request_cpu=1,
    limit_cpu=4,
    request_mem=500,
    limit_mem=1000)
print("\n")
cluster.create_pod(
    name='directly-created-pod',
    image='nginx',
    labels=None,
    namespace=cluster.namespace,
    request_mem="{}Mi".format(200),
    request_cpu=str(0.5),
    limit_mem="{}Mi".format(400),
    limit_cpu=str(1))
sleep(5)
print("\n")
print(f"pods in the cluster: {cluster.existing_pods}")
print("\n")
print(f"all pods metrics:\n{cluster.get_pods_metrics()}")
print("\n")
print("experimental-pod metrics: \n")
print(f"{cluster.get_pod_metrics('experimental-pod')}")
print("\n")
print("'directly-created-pod metrics': \n")
print(f"{cluster.get_pod_metrics('directly-created-pod')}")
sleep(5)
cluster.delete_pod(name='experimental-pod')
print("\n")
cluster.delete_pod(name='directly-created-pod')
print("\n")
# --------------- cleaning cluster ---------------
cluster.clean()
