"""scripts to check the functionalities of the environment
   for pod creation
"""

from smart_kube import Cluster
from time import sleep
from smart_kube.cluster import clean_all_namespaces


clean_all_namespaces()
# --------------- cluster operations ---------------
cluster = Cluster(namespace="deployment-operations")
print("\n")

# --------------- deployments operations ---------------
cluster.create_experiment_deployment(
    deployment_name="experimental-deployment",
    replicas=1,
    request_mem=100,
    request_cpu=0.5,
    limit_mem=200,
    limit_cpu=1,
)
print("\n")

cluster.create_deployment(
    deployment_name="directly-created-deployment",
    image="nginx",
    replicas=2,
    deployment_selector={"app": "nginx"},
    pod_labels={"app": "nginx"},
    containers_name="directly-created-deployment",
    request_mem="100Mi",
    request_cpu="0.5",
    limit_mem="200Mi",
    limit_cpu="1",
)
print("\n")

sleep(5)
cluster.update_deployment(
    deployment_name="experimental-deployment", limit_mem="500Mi")
print("\n")

sleep(5)
cluster.delete_deployment(name="experimental-deployment")
print("\n")
cluster.delete_deployment(name="directly-created-deployment")
print("\n")

# --------------- cleaning cluster ---------------
cluster.clean()
