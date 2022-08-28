import os
from plistlib import load
from re import TEMPLATE
from typing import Any, Dict
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from datasets import load_dataset
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-audio-qa/seldon-core-version"
CHECK_TIMEOUT = 2
RETRY_TIMEOUT = 60
DELETE_WAIT = 10
TEMPLATE = "audio"


ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

inputs=ds[0]["audio"]["array"]

def load_test(
    pipeline_name: str,
    inputs: Dict[str, Any]
    ):
    # TODO change here
    # single node inferline
    gateway_endpoint="localhost:32000"
    deployment_name = pipeline_name 
    namespace = "default"

    sc = SeldonClient(
        gateway_endpoint=gateway_endpoint,
        gateway="istio",
        transport="rest",
        deployment_name=deployment_name,
        namespace=namespace)

    time.sleep(CHECK_TIMEOUT)
    response = sc.predict(
        data=inputs
    )

    if response.success:
        pp.pprint(response.response['jsonData'])
    else:
        pp.pprint(response.msg)

def setup_pipeline(
    node_1_model: str,
    node_2_model: str, 
    template: str,
    pipeline_name: str):
    svc_vars = {
        "node_1_variant": node_1_model,
        "node_2_variant": node_2_model,        
        "pipeline_name": pipeline_name}
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template(f"{template}.yaml")
    content = svc_template.render(svc_vars)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
    """
    os.system(command)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")

# check all the possible combination
node_1_models = [
    'facebook/s2t-small-librispeech-asr']

node_2_models = [
    'distilbert-base-uncased-finetuned-sst-2-english']

for node_1_model in node_1_models:
    for node_2_model in node_2_models:
        pipeline_name = node_1_model[:8].lower() + "-" +\
            node_2_model[:8].lower()
        start_time = time.time()
        while True:
            setup_pipeline(
                node_1_model=node_1_model,
                node_2_model=node_2_model,
                template=TEMPLATE, pipeline_name=pipeline_name)
            time.sleep(CHECK_TIMEOUT)
            command = ("kubectl rollout status deploy/$(kubectl get deploy"
                    f" -l seldon-deployment-id={pipeline_name} -o"
                    " jsonpath='{.items[0].metadata.name}')")
            time.sleep(CHECK_TIMEOUT)
            p = subprocess.Popen(command, shell=True)
            try:
                p.wait(RETRY_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                p.kill()
                print("corrupted pipeline, should be deleted ...")
                remove_pipeline(pipeline_name=pipeline_name)
                print('waiting to delete ...')
                time.sleep(DELETE_WAIT)

        print('starting the load test ...\n')
        load_test(pipeline_name=pipeline_name, inputs=inputs)

        time.sleep(DELETE_WAIT)

        print("operation done, deleting the pipeline ...")
        remove_pipeline(pipeline_name=pipeline_name)
        print('pipeline successfuly deleted')
