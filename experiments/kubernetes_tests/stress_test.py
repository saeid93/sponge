from smart_kube import Cluster
from smart_kube.cluster import clean_all_namespaces

clean_all_namespaces()
# Cluster Configurations
# CONFIG_FILE = '~/.kube/config'
# TODO add the workload loading of check-vpa here
NAMESPACE = 'vpa'
WORKLPAD_PATH = 'workload.pickle'
UTILIZATION_IMAGE = 'sdghafouri/stress-utilization-server'


# Pod Configurations
POD_SVC_NAME = 'sample-vpa'
POD_NAME = 'sample-vpa'
POD_IMAGE = 'sdghafouri/stress-pod'


# create Cluster and create utilization-server
cluster = Cluster(
    namespace=NAMESPACE
)

cluster.setup_utilization_server(
    image=UTILIZATION_IMAGE,
    workloads_path=WORKLPAD_PATH)

cluster.create_pod(
    labels={'svc': POD_NAME},
    name=POD_NAME,
    image=POD_IMAGE)

cluster.create_service(
    name=POD_SVC_NAME)
