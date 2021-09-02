from time import sleep

from kubernetes.client.rest import ApiException
from kubernetes.client import CoreV1Api
from kubernetes import config

from smart_kube.util import logger


def clean_all_namespaces(config_file_path: str = "~/.kube/config"):
    config.load_kube_config(config_file_path)
    core_api: CoreV1Api = CoreV1Api()
    response = core_api.list_namespace()
    namespaces = list(map(lambda a: a.metadata.name, response.items))
    builtin_namespaces = [
        "default", "kube-node-lease", "kube-public", "kube-system",
        "monitoring"]
    user_defined_namespaces = list(set(namespaces) - set(builtin_namespaces))
    logger.info(
        "user defined namespaces in the cluster {}".format(
            user_defined_namespaces)
    )
    for namespace in user_defined_namespaces:
        logger.info("removing namespace <{}>".format(namespace))
        core_api.delete_namespace(namespace)
        while True:
            sleep(1)
            try:
                core_api.read_namespace(namespace)
            except ApiException:
                logger.warn("namespace <{}> removed.".format(namespace))
                break
    logger.info("all user defined namespaces removed")
