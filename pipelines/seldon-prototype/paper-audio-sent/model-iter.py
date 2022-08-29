"""
Iterate through all possible combination
of models and servers
"""

import os
from plistlib import load
import yaml
from re import TEMPLATE
from typing import Any, Dict
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from datasets import load_dataset
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-audio-sent/seldon-core-version"
PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline-test-meta" # TODO fix be moved to utilspr
DATABASE = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline"
CHECK_TIMEOUT = 2 
RETRY_TIMEOUT = 90
DELETE_WAIT = 45
LOAD_TEST_WAIT = 60
TRIAL_END_WAIT = 60
TEMPLATE = "audio"
CONFIG_FILE = "paper-audio-sent"


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

config_file_path = os.path.join(
    PIPELINES_MODELS_PATH, f"{CONFIG_FILE}.yaml")
with open(config_file_path, 'r') as cf:
    config = yaml.safe_load(cf)

node_1_models = config['node_1']
node_2_models = config['node_2']

def prune_name(name, len):
    forbidden_strs = ['facebook', '/', 'huggingface', '-']
    for forbidden_str in forbidden_strs:
        name = name.replace(forbidden_str, '')
    name = name.lower()
    name = name[:len]
    return name

for node_1_model in node_1_models:
    for node_2_model in node_2_models:
        pipeline_name = prune_name(node_1_model, 8) + "-" +\
            prune_name(node_2_model, 8)
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
