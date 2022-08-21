import os
from plistlib import load
from typing import Any, Dict
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from itertools import islice

PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/20-motivational-tests/video-monitoring/seldon-core-version"
CHECK_TIMEOUT = 2
RETRY_TIMEOUT = 60
DELETE_WAIT = 30

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
                :num_loaded_images]}
    return images

def load_test(
    pipeline_name: str,
    images: Dict[str, Any],
    n_items: int
    ):
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

    images = dict(islice(images.items(), n_items))
    results = {}
    for image_name, image in images.items():
        image = np.array(image)
        response = sc.predict(
            data=image
        )
        results[image_name] = response

    for image_name, response in results.items():
        print(f"\nimage name: {image_name}")
        print(f"-"*50)
        if response.success:
            request_path = response.response['meta']['requestPath'].keys()
            pipeline_response = response.response['data']
            print(f"request path: {request_path}")
            print(f"pipeline_response: {pipeline_response}")
        else:
            print(f"{image_name} -> {response.msg}")

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

images = load_images(num_loaded_images=10)

# check all the possible combination


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

        print('\starting the load test ...\n')
        load_test(pipeline_name=pipeline_name, images=images, n_items=1)

        time.sleep(30)

        print("operation done, deleting the pipeline ...")
        remove_pipeline(pipeline_name=pipeline_name)
        print('waiting to delete ...')
