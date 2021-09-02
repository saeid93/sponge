import kopf
from kubernetes import client, config


config.load_kube_config()
custom_api = client.CustomObjectsApi()
api = client.CoreV1Api()


def get_pod_labels(config_map_name="flaskapp_config"):
    """extract pod labels from config monitor"""
    results = []

    # get all crds
    response = custom_api.list_cluster_custom_object(
        group="magalix.com",
        version="v1",
        plural="configmonitors"
    )

    for r in response['items']:
        if r['spec']['configmap'] == config_map_name:
            results.append(
                r['spec']['podSelector']
            )

    results = [list(labels.keys())[0] + "=" + labels[list(labels.keys())[0]]
               for labels in results]

    return results


def kill_pod_by_label(labels="", namespace="default"):
    """delete pods by label"""
    for label in labels:
        response = api.list_namespaced_pod(
            namespace=namespace,
            label_selector=label
            )

        for r in response.items:
            api.delete_namespaced_pod(
                name=r.metadata.name,
                namespace=namespace
            )


@kopf.on.event('', 'v1', 'configmaps')
def cm_events(type, meta, spec, **kwargs):
    # get the name of the configmap
    name = meta.get('name')

    # if it is flaskapp-config and it is modified restart the pod
    if name == "flaskapp-config" and type == "MODIFIED":
        # delete by pod labels
        labels = get_pod_labels(name)
        kill_pod_by_label(labels)
