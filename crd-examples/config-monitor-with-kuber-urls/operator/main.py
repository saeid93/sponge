import requests
import os
import json
import logging
import sys

log = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
log.addHandler(out_hdlr)
log.setLevel(logging.INFO)


base_url = "http://127.0.0.1:8001"


namespace = os.getenv("res_namespace", "default")

# This is the function that searches for and kills Pods by searching for
# them by label


def kill_pods(labels):
    # We receive labels in the form of a list
    for label in labels:
        url = "{}/api/v1/namespaces/{}/pods?labelSelector={}".format(
            base_url, namespace, label)
        r = requests.get(url)
        # Make the request to the endpoint to retreive the Pods
        response = r.json()
        # Extract the Pod name from the list
        pods = [p['metadata']['name'] for p in response['items']]
        # For each Pod, issue an HTTP DELETE request
        for p in pods:
            url = "{}/api/v1/namespaces/{}/pods/{}".format(
                base_url, namespace, p)
            r = requests.delete(url)
            if r.status_code == 200:
                log.info("{} was deleted successfully".format(p))
            else:
                log.error("Could not delete {}".format(p))


# This function is used to extract the Pod labels
# from the configmonitor resource.
# It takes the configmap name as the argument and
# uses it to search for configmonitors
# that have the configmap name in its spec
def getPodLabels(configmap):
    url = "{}/apis/magalix.com/v1/namespaces/{}/configmonitors".format(
        base_url, namespace)
    r = requests.get(url)
    # Issue the HTTP request to the appropriate endpoint
    response = r.json()
    # Extract the podSelector part from each object in the response
    pod_labels_json = [i['spec']['podSelector']
                       for i in response['items'] if i[
                           'spec']['configmap'] == "flaskapp-config"]
    print(pod_labels_json)
    result = [list(labels.keys())[0] + "=" + labels[list(labels.keys())[0]]
              for labels in pod_labels_json]
    # The result is a list of labels
    return result

# This is the main function that watches the API for changes


def event_loop():
    log.info("Starting the service")
    url = '{}/api/v1/namespaces/{}/configmaps?watch=true"'.format(
        base_url, namespace)
    r = requests.get(url, stream=True)
    # We issue the request to the API endpoint and keep the conenction open
    for line in r.iter_lines():
        obj = json.loads(line)
        # We examine the type part of the object to see if it is MODIFIED
        event_type = obj['type']
        # and we extract the configmap name because we'll need it later
        configmap_name = obj["object"]["metadata"]["name"]
        if event_type == "MODIFIED":
            log.info("Modification detected")
            # If the type is MODIFIED then we extract the pod labels by using
            # the getPodLabels function
            # passing the configmap name as a parameter
            labels = getPodLabels(configmap_name)
            # Once we have the labels, we can use them to find and kill the
            # Pods by calling the
            # kill_pods function
            print(f"{labels}")
            kill_pods(labels)


event_loop()
