# %%
import os

#
# %%
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from torchvision import transforms
import threading
import pandas as pd
import math
import os
from PIL import Image
import transformers
import torch
from time import sleep
import time
import multiprocessing as mp
from prom import *
import requests

data = []
pod = "triton1-678549f99c-l9ff5"
name_space = "default"
database = "profile-exp8-text/1/"
num_requests = [120, 90, 50, 30, 20, 15]
url = 30803
def send_request(model_name, model_version, inputs, outputs, batch_size):
    global url
    start_load = time.time()
    res = requests.post(url=f'http://localhost:{url}/v2/repository/models/{model_name}/load')
    load_time = time.time() - start_load
      

    with open(database+"load-time.txt", "a") as f:
        f.write(f"load time of {model_name} is {load_time} \n")

    sleep(60)
    cpu_usages = []
    memory_usages = []
    infer_times = []
    input_times = []
    output_times = []
    queue_times = []
    success_times = []

  
    
    print(model_name, model_version, "start")
    for i in range(170):
        try:
            triton_client = httpclient.InferenceServerClient(
                url=f'localhost:{url}'
            )
        except Exception as e:
            print("context creation failed: " + str(e))

        start_time = time.time()
        result = triton_client.infer(
                    model_name=model_name,model_version=model_version, inputs=inputs, outputs=outputs)
        latency = time.time() - start_time

        triton_client.close()
        print(i)

        data.append([i, model_name, model_version,batch_size, latency])
        
        minutes = 1
        if i > 10:
            cpu_usages.append(get_cpu_usage(pod, name_space, minutes, minutes))
            memory_usages.append(get_memory_usage(pod, name_space, minutes, minutes))
            infer_times.append(get_inference_duration(model_name, model_version, pod))
            queue_times.append(get_queue_duration(model_name, model_version, pod))
            success_times.append(get_inference_count(model_name, model_version, pod))
    
    end_time = 5
   
    sleep(70)
    total_time = time.time() - start_load
    minutes = total_time // 60
    minutes = int(minutes)
    if minutes < 2:
        minutes = 2
    end_infer = 0
    if minutes < 10:
        end_infer = 10
    
    else:
        end_infer = minutes + 5


    cpu_usages.append(get_cpu_usage(pod, name_space, minutes, minutes))
    memory_usages.append(get_memory_usage(pod, name_space, minutes, minutes))
    infer_times.append(get_inference_duration(model_name, model_version, pod))
    success_times.append(get_inference_count(model_name, model_version, pod))
    queue_times.append(get_queue_duration(model_name, model_version, pod))

    with open(database+"cpu.txt", "a") as cpu_file:
        cpu_file.write(f"usage of {model_name} {model_version} on batch {batch_size} is {cpu_usages} \n")

    with open(database+"memory.txt", 'a') as memory_file:
        memory_file.write(f"usage of {model_name} {model_version} on batch {batch_size} is {memory_usages} \n")

    with open(database+"infer-prom.txt", "a") as infer:
        infer.write(f"infertime of {model_name} {model_version} on batch {batch_size} is {infer_times} \n")
    
    with open(database+"queue_times.txt", 'a') as q:
        q.write(f"Queuetimes of {model_name} {model_version} on batch {batch_size} is {queue_times} \n")

    with open(database+"success.txt", "a") as s:
        s.write(f"success of {model_name} {model_version} on batch {batch_size} is {success_times} \n")

    requests.post(url=f'http://localhost:{url}/v2/repository/models/{model_name}/unload')


        


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


def create_input(inp):
    inputs = []
    for k in inp.keys():
        pass


@click.command()
@click.option('--config-file', type=str, default='temp')
def main(config_file: str):
    print("sleep for one minute to heavy start")
    sleep(25)
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
    print(model_versions)

    results = []
    processes = []

    inputs = []
    for bat in [2, 4, 8, 16, 32]:
        print(f"start batch 1")
        
        for j,model_name in enumerate(model_names):
            for version in model_versions[j]:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inp = tokenizer(["This is a sample" for _ in range(bat)], return_tensors="pt")
                inputs = []
                inputs.append(
                    httpclient.InferInput(
                        name="input_ids",shape=inp['input_ids'].shape, datatype="INT64"
                )
                )
                inputs[0].set_data_from_numpy(inp['input_ids'].numpy(), binary_data=False)
                
                inputs.append(
                    httpclient.InferInput(
                        name="attention_mask", shape=inp['attention_mask'].shape, datatype="INT64")
                )
                inputs[1].set_data_from_numpy(inp['attention_mask'].numpy())
                
                outputs = []
                outputs.append(httpclient.InferRequestedOutput(name="logits"))
    

                model_name_s = model_name
                if "/" in model_names:
                    model_name_s = model_names.replace("/","")
                
                send_request(model_name_s, version, inputs, outputs, bat)

    sleep(120)
    df = pd.DataFrame(columns=['index', 'model-name', 'model-version', 'batch-size', 'latency'])
    for i, m, v, b, l in data:
        df.loc[len(df)] = [i, m, v, b, l]

    df.to_csv(database+"data.csv")

if __name__ == "__main__":
    main()





# # %%
# # use onnx model




