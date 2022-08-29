from PIL import Image
import transformers
import torch
from time import sleep
import time
import multiprocessing as mp
from prom import *
import requests
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from torchvision import transforms
import threading
import pandas as pd
import math

data = []
pod = "triton-6c66d4b4df-vzs59"
name_space = "default"
database = "profile-exp6-cores/8-new/"
num_requests = [120, 90, 50, 30, 20, 15]
url = "30900"
def send_request(model_name, model_version):
    global url
    start_load = time.time()
    res = requests.post(url=f'http://localhost:{url}/v2/repository/models/{model_name}/load')
    load_time = time.time() - start_load
    os.system(f"perf_analyzer -m {model_name} -b 2 --shape IMAGE:3,224,224 -t1 -v -p 10000 -u localhost:{url} -f {database}/{model_name}-{model_version}.csv")
    res = requests.post(url=f'http://localhost:{url}/v2/repository/models/{model_name}/unload')



import click
import sys
import yaml
import os
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..')))

from utilspr.constants import (
    TEMP_MODELS_PATH,
    KUBE_YAMLS_PATH
    )

@click.command()
@click.option('--config-file', type=str, default='model-load')
def main(config_file: str):
    print("sleep for one minute to heavy start")
    sleep(10)
    config_file_path = os.path.join(
        KUBE_YAMLS_PATH, f"{config_file}.yaml")
    with open(config_file_path, 'r') as cf:
        config = yaml.safe_load(cf)

    model_names = config['model_names']
    versions = config['versions']
    model_versions = [[] for _ in range(len(versions))]
    for k, version in enumerate(versions):
        for i in range(len(version)):
            model_versions[k].append(str(i+1))
    
    for j,model_name in enumerate(model_names):
            for version in model_versions[j]:
                send_request(model_name, version)
    
if __name__ == "__main__":
    main()


                