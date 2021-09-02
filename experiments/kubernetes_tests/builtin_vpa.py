"""scripts to check the functionalities of the environment
"""

from smart_kube import Cluster
from time import sleep
from smart_kube.cluster import clean_all_namespaces


clean_all_namespaces()
# --------------- cluster operations ---------------
cluster = Cluster(namespace="builtin-vpa-operations")
cluster = Cluster(namespace="default")
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


# --------------- vpa operations ---------------
cluster.activate_builtin_vpa(
    deployment_name="experimental-deployment", update_mode="Off"
)
print("\n")

cluster.get_builin_vpa_recommendation(
    deployment_name="experimental-deployment")
print("\n")

print(f"deploymentes with autoscalers:\n{cluster.existing_vpas}")
print("\n")
sleep(5)  # wait for the autoscaler recommendations to get activated
for i in range(10):
    sleep(10)
    print("----- vpa recomms -----")
    print(
        cluster.get_builin_vpa_recommendation(
            deployment_name="experimental-deployment")
    )
    print("----- vpa exp recomms -----")
    print(
        cluster.get_builin_vpa_recommendation_experimental_deployment(
            deployment_name="experimental-deployment"
        )
    )
    print("\n")

sleep(10)
cluster.delete_deployment(name="experimental-deployment")
print("\n")
cluster.delete_deployment(name="directly-created-deployment")
print("\n")

# --------------- cleaning cluster ---------------
cluster.clean()
