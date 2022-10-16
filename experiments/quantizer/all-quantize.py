import os
from re import TEMPLATE
import yaml
from typing import Any, Dict
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-nlp/seldon-core-version"
PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline-test-meta" # TODO fix be moved to utilspr
DATABASE = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline"
CHECK_TIMEOUT = 60
RETRY_TIMEOUT = 90
DELETE_WAIT = 45
LOAD_TEST_WAIT = 60
TRIAL_END_WAIT = 60
TEMPLATE = "nlp"
CONFIG_FILE = "paper-nlp"

config_file_path = os.path.join(
    PIPELINES_MODELS_PATH, f"{CONFIG_FILE}.yaml")
with open(config_file_path, 'r') as cf:
    config = yaml.safe_load(cf)

node_1_models = config['node_3']
for model in node_1_models:
    file = model.replace("/","-")
    command = f""" python text-quantizer.py --model_name_or_path {model} --task_name sst2  --quantization_approach dynamic  --do_eval  --output_dir ./{file} """
    try:
        os.system(command)
    except:
        pass
