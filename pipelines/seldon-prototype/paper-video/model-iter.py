"""
Iterate through all possible combination
of models and servers
"""

import os
from plistlib import load
from typing import Any, Dict
from PIL import Image
from prom import get_cpu_usage, get_memory_usage
import numpy as np
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from itertools import islice
from torchvision import transforms
import pprint
PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-video/seldon-core-version"
CHECK_TIMEOUT = 2
RETRY_TIMEOUT = 60
DELETE_WAIT = 15

def load_images(num_loaded_images = 10):
    os.system('sudo umount -l ~/my_mounting_point')
    os.system('cc-cloudfuse mount ~/my_mounting_point')

    data_folder_path = '/home/cc/my_mounting_point/datasets'
    dataset_folder_path = os.path.join(
        data_folder_path, 'ILSVRC/Data/DET/test'
    )
    classes_file_path = os.path.join(
        data_folder_path, 'imagenet_classes.txt'
    )

    image_names = os.listdir(dataset_folder_path)
    image_names.sort()
    with open(classes_file_path) as f:
        classes = [line.strip() for line in f.readlines()]

    def image_loader(folder_path, image_name):
        image = Image.open(
            os.path.join(folder_path, image_name))
        # if there was a need to filter out only color images
        # if image.mode == 'RGB':
        #     pass
        return image
    images = {
        image_name: image_loader(
            dataset_folder_path, image_name) for image_name in image_names[
                1:num_loaded_images+1]}
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
    return images

def load_test(
    pipeline_name: str,
    images: Dict[str, Any],
    n_items: int,
    n_iters:int
    ):
    # TODO change here
    # single node inferline
    gateway_endpoint="localhost:32000"
    deployment_name = pipeline_name 
    namespace = "default"
    start = time.time()
    sc = SeldonClient(
        gateway_endpoint=gateway_endpoint,
        gateway="istio",
        transport="rest",
        deployment_name=deployment_name,
        namespace=namespace)

    images = dict(islice(images.items(), n_items))
    # print(images)
    results = {}
    cpu_usages = []
    memory_usages = []
    e_e_lats = []
    yolo_lats = []
    resnet_lats = []
    print(f"start load test on {pipeline_name} after 60 seconds ")
    time.sleep(60)
    for i in range(n_iters):
        # print("hiiiiiiiiiiiii")
        print(i)
        for image_name, image in images.items():
            response = sc.predict(
                data=np.array(image)
            )

            results[image_name] = response

        for image_name, response in results.items():
            if response.success:
                request_path = response.response['meta']['requestPath'].keys()
                if 'jsonData' in response.response.keys(): 
                    pipeline_response = response.response['jsonData']
                    times = pipeline_response['time']
                    arrival_yolo = times['arrival_video-yolo']
                    arrival_resnet = times['arrival_video-resnet-human']
                    serving_yolo = times['serving_video-yolo']
                    serving_resnet = times['serving_video-resnet-human']
                    end_to_end_latency = serving_resnet - arrival_yolo
                    yolo_latency = serving_yolo - arrival_yolo
                    resnet_latency = serving_resnet - arrival_resnet
                    e_e_lats.append(end_to_end_latency)
                    yolo_lats.append(yolo_latency)
                    resnet_lats.append(resnet_latency)

                    cpu_usages.append(get_cpu_usage(pipeline_name, "default"))
                    memory_usages.append(get_memory_usage(pipeline_name, "default", 0))

                else:
                    pipeline_response = response.response['data']
    sc.close()
    total_time = int((time.time() - start)//60)
    time.sleep(70)
    cpu_usages.append(get_cpu_usage(pipeline_name, "default"))
    memory_usages.append(get_memory_usage(pipeline_name, "default", total_time, True)) 

    database = ""
    with open(database+"cpu.txt", "a") as cpu_file:
        cpu_file.write(f"usage of {pipeline_name} is {cpu_usages} \n")

    with open(database+"memory.txt", 'a') as memory_file:
        memory_file.write(f"usage of {pipeline_name} is {memory_usages} \n")

    with open(database+"yolo.txt", "a") as infer:
        infer.write(f"yololats of {pipeline_name} is {yolo_lats} \n")
    
    with open(database+"resnet.txt", 'a') as q:
        q.write(f"resnetlats of {pipeline_name} is {resnet_lats} \n")

    with open(database+"ee.txt", "a") as s:
        s.write(f"eelat of {pipeline_name} is {e_e_lats} \n")
    


def setup_pipeline(
    yolo_variant: str,
    resnet_variant: str,
    template: str,
    pipeline_name: str):
    svc_vars = {
        "yolo_variant": yolo_variant,
        "resnet_variant": resnet_variant,
        "pipeline_name": pipeline_name}
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template(f"{template}.yaml")
    config_path = os.path.join(PATH, f"{template}.yaml")
    content = svc_template.render(svc_vars)
    with open(config_path, mode="w", encoding="utf-8") as message:
        message.write(content)
    os.system(f"kubectl apply -f {config_path} -n default")
    os.remove(config_path)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")

images = load_images(num_loaded_images=1)
n_iters = 100
# check all the possible combination
print("images loaded")
print("-"*50)

resnet_models = [
    'resnet18', 'resnet34', 'resnet50',
    'resnet101', 'resnet152']

yolo_models = [
    'yolov5n', 'yolov5s', 'yolov5m',
    'yolov5l', 'yolov5x', 'yolov5n6',
    'yolov5s6', 'yolov5m6', 'yolov5l6',
    'yolov5l6']

for resnet_model in resnet_models:
    for yolo_model in yolo_models:
        pipeline_name = yolo_model + resnet_model
        start_time = time.time()
        while True:
            setup_pipeline(
                yolo_variant=yolo_model, resnet_variant=resnet_model,
                template="video-monitoring", pipeline_name=pipeline_name)
            command = ("kubectl rollout status deploy/$(kubectl get deploy"
                       f" -l seldon-deployment-id={pipeline_name} -o"
                       " jsonpath='{.items[0].metadata.name}')")
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
                continue
        time.sleep(20)
        print('\starting the load test ...\n')
        load_test(pipeline_name=pipeline_name, images=images, n_items=10, n_iters=n_iters)

        time.sleep(30)
        print("operation done, deleting the pipeline ...")
        remove_pipeline(pipeline_name=pipeline_name)
        print('waiting to delete ...')
