# %%
print("start")
import os
from PIL import Image
import transformers
import torch
from time import sleep
import time
import multiprocessing as mp
from prom import *
import requests
# %%
# !kubectl create secret generic aws-credentials --from-literal=AWS_ACCESS_KEY_ID=minioadmin --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin

# %%
# %%writefile triton-deploy.yaml
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   labels:
#     app: triton
#   name: triton
# spec:
#   replicas: 1
#   selector:
#     matchLabels:
#       app: triton
#   template:
#     metadata:
#       labels:
#         app: triton
#     spec:
#       containers:
#       - image: nvcr.io/nvidia/tritonserver:21.09-py3
#         name: tritonserver
#         command: ["/bin/bash"]
#         args: ["-c", "cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates && update-ca-certificates && /opt/tritonserver/bin/tritonserver --model-store=s3://http://minio.minio-system.svc.cluster.local:9000/minio-seldon/models --strict-model-config=false"]
#         env:
#         - name: AWS_ACCESS_KEY_ID
#           valueFrom:
#             secretKeyRef:
#               name: aws-credentials
#               key: AWS_ACCESS_KEY_ID
#         - name: AWS_SECRET_ACCESS_KEY
#           valueFrom:
#             secretKeyRef:
#               name: aws-credentials
#               key: AWS_SECRET_ACCESS_KEY      
#         ports:
#           - containerPort: 8000
#             name: http
#           - containerPort: 8001
#             name: grpc
#           - containerPort: 8002
#             name: metrics
#         volumeMounts:
#         - mountPath: /dev/shm
#           name: dshm
#       volumes:
#       - name: dshm
#         emptyDir:
#           medium: Memory

# # %%
# %%writefile triton-service.yaml
# apiVersion: v1
# kind: Service
# metadata:
#   name: triton
# spec:
#   type: NodePort
#   selector:
#     app: triton
#   ports:
#     - protocol: TCP
#       name: http
#       port: 8000
#       nodePort: 30800
#       targetPort: 8000
#     - protocol: TCP
#       name: grpc
#       port: 8001
#       nodePort: 30801
#       targetPort: 8001
#     - protocol: TCP
#       name: metrics
#       nodePort: 30802
#       port: 8002
#       targetPort: 8002

# # %%
# model_names = ['resnet-18', 'resnet-50', 'vit-base-32', 'vit-base-64']

# # %%
# for model in model_names:
#     os.system(f'python -m transformers.onnx --model={model} models/{model}/1 ')

# # %%
# %%writefile triton-deploy.yaml
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   labels:
#     app: triton
#   name: triton
# spec:
#   replicas: 1
#   selector:
#     matchLabels:
#       app: triton
#   template:
#     metadata:
#       labels:
#         app: triton
#     spec:
#       containers:
#       - image: nvcr.io/nvidia/tritonserver:21.09-py3
#         name: tritonserver
#         command: ["/bin/bash"]
#         args: ["-c", "cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates && update-ca-certificates && /opt/tritonserver/bin/tritonserver --model-store=s3://http://minio.minio-system.svc.cluster.local:9000/minio-seldon/models --strict-model-config=false"]
#         env:
#         - name: AWS_ACCESS_KEY_ID
#           valueFrom:
#             secretKeyRef:
#               name: aws-credentials
#               key: AWS_ACCESS_KEY_ID
#         - name: AWS_SECRET_ACCESS_KEY
#           valueFrom:
#             secretKeyRef:
#               name: aws-credentials
#               key: AWS_SECRET_ACCESS_KEY      
#         ports:
#           - containerPort: 8000
#             name: http
#           - containerPort: 8001
#             name: grpc
#           - containerPort: 8002
#             name: metrics
#         volumeMounts:
#         - mountPath: /dev/shm
#           name: dshm
#       volumes:
#       - name: dshm
#         emptyDir:
#           medium: Memory

# # %%
# %%writefile triton-service.yaml
# apiVersion: v1
# kind: Service
# metadata:
#   name: triton
# spec:
#   type: NodePort
#   selector:
#     app: triton
#   ports:
#     - protocol: TCP
#       name: http
#       port: 8000
#       nodePort: 30800
#       targetPort: 8000
#     - protocol: TCP
#       name: grpc
#       port: 8001
#       nodePort: 30801
#       targetPort: 8001
#     - protocol: TCP
#       name: metrics
#       nodePort: 30802
#       port: 8002
#       targetPort: 8002

# %%
def create_batch(batch_size):
    batch = []
    for i in range(batch_size):
        batch.append(read_file())
    return batch

# %%
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

# %%
def create_batch_image(batch_size):
    num_loaded_images = batch_size
    images = {
        image_name: image_loader(
            dataset_folder_path, image_name) for image_name in image_names[
                :num_loaded_images]}
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
 
    return torch.stack(list(map(lambda a: transform(a), list(images.values()))))


# %%
class Profiler:
    def __init__(self, model_name, batch):
        self.model_name = model_name
        self.batch = batch
        try:
            triton_client = httpclient.InferenceServerClient(
                url='localhost:30800'
            )
        except Exception as e:
            print("context creation failed: " + str(e))
        inputs = []
        inputs.append(
            httpclient.InferInput(
                name="input", shape=batch.shape, datatype="FP32")
        )
        inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput(name="output"))
 
            
    def runner(self, counter):
        for i in range(counter):
            result = triton_client.infer(
            model_name=model_name, inputs=encoded_input, outputs=outputs)
            triton_client.close()
            


            

# %%
# results = [[] for i in range(len(model_names))]
# for i,model in enumerate(model_names):
#     for batch in [1, 2, 4, 16]:
#         p = Profiler(model, create_batch_images(batch))
#         p.runner()
#         results[i].append(requests.get("localhost:8003/metrics"))

# %%
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from torchvision import transforms
import threading
import pandas as pd

data = []
def send_request(model_name, model_version, inputs, outputs, batch_size):
    start_load = time.time()
    requests.post(url=f'http://localhost:30800/v2/repository/models/{model_name}/load')
    load_time = time.time() - start_load
    with open("load-time.txt", "a") as f:
        f.write(f"load time of {model_name} is {load_time} \n")

    
    print(model_name, model_version, "start")
    for i in range(20):
        try:
            triton_client = httpclient.InferenceServerClient(
                url='localhost:30800'
            )
            start_time = time.time()
            result = triton_client.infer(
                        model_name=model_name,model_version=model_version, inputs=inputs, outputs=outputs)
            triton_client.close()
            latency = time.time() - start_time

            data.append([i, model_name, model_version,batch_size, latency])
            print(i)
        except Exception as e:
            print("context creation failed: " + str(e))
    sleep(60)
    cpu_usage = get_cpu_usage("triton-67dff8d668-6qstk")
    memory_usage = get_memory_usage("triton-67dff8d668-6qstk")
    with open("cpu.txt", "a") as cpu_file:
        cpu_file.write(f"usage of {model_name} {model_version} on batch {batch_size} is {cpu_usage} \n")

    with open("memory.txt", 'a') as memory_file:
        memory_file.write(f"usage of {model_name} {model_version} on batch {batch_size} is {memory_usage} \n")
    sleep(5)
    requests.post(url=f'http://localhost:30800/v2/repository/models/{model_name}/unload')


        

import click
import sys
import yaml
import os
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..')))

from utils.constants import (
    TEMP_MODELS_PATH,
    KUBE_YAMLS_PATH
    )

@click.command()
@click.option('--config-file', type=str, default='model-load')
def main(config_file: str):
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
    for bat in [2, 4, 8, 16, 32, 64]:
        os.system('sudo umount -l ~/my_mounting_point')
        os.system('cc-cloudfuse mount ~/my_mounting_point')
        inputs = []

        print(f"start batch {bat}")
        batch =create_batch_image(bat)
        inputs.append(
                        httpclient.InferInput(
                            name="input", shape=batch.shape, datatype="FP32")
                    )
        inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(name="output"))
        for j,model_name in enumerate(model_names):
            for version in model_versions[j]:
                # p = mp.Process(target=send_request, args=(model_name,version,inputs,outputs,))
                # p.start()
                # processes.append(p)
                send_request(model_name, version, inputs, outputs, bat)
                sleep(120)

    sleep(120)
    df = pd.DataFrame(columns=['index', 'model-name', 'model-version', 'batch-size', 'latency'])
    for i, m, v, b, l in data:
        df.loc[len(df)] = [i, m, v, b, l]

    df.to_csv("data.csv")

if __name__ == "__main__":
    main()


                
                



