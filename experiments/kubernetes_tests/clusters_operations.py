"""scripts to check the functionalities of the environment
   for cluster operations
"""

from smart_kube import Cluster
from smart_kube.cluster import clean_all_namespaces


clean_all_namespaces()
# --------------- cluster operations ---------------
cluster = Cluster(namespace='pods-operations')
print("\n")
# --------------- cleaning cluster ---------------
cluster.clean()
