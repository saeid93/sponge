from kubernetes import client, config, watch
import logging
import sys


log = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
log.addHandler(out_hdlr)
log.setLevel(logging.INFO)


class ConfigMonitorOperator:
    def __init__(self, namespace="default") -> None:
        self.api = client.CoreV1Api()
        self.namespace = namespace
        self.custom_api = client.CustomObjectsApi()

    def kill_pod_by_label(self, labels=""):
        """delete pods by label"""
        for label in labels:
            response = self.api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label
                )

            for r in response.items:
                self.api.delete_namespaced_pod(
                    name=r.metadata.name,
                    namespace=self.namespace
                )
                log.info(f"{r.metadata.name} is deleted successfully")

    def get_pod_labels(self, config_map_name="flaskapp_config"):
        """extract pod labels from config monitor"""
        results = []

        # get all crds
        response = self.custom_api.list_cluster_custom_object(
            group="magalix.com",
            version="v1",
            plural="configmonitors"
        )

        for r in response['items']:
            if r['spec']['configmap'] == config_map_name:
                results.append(
                    r['spec']['podSelector']
                )

        results = [list(labels.keys())[0] + "=" + labels[
            list(labels.keys())[0]]
                   for labels in results]

        return results

    def event_loop(self):
        """Watch the configmap events and adapt them"""
        log.info("starting the service")
        watch_ = watch.Watch()
        for event in watch_.stream(
            func=self.api.list_namespaced_config_map,
            namespace=self.namespace
                ):
            configmap_name = event['object'].metadata.name

            if event['type'] == "MODIFIED":
                log.info("Modification is detected!")
                labels = self.get_pod_labels(configmap_name)
                self.kill_pod_by_label(labels=labels)


if __name__ == "__main__":
    config.load_kube_config()
    operator = ConfigMonitorOperator()
    operator.event_loop()
