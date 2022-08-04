# %%
print("alireza")
import os
from PIL import Image
import transformers
import torch
from time import sleep
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

def send_request(model_name, model_version):
    try:
        triton_client = httpclient.InferenceServerClient(
            url='localhost:30800'
        )
    except Exception as e:
        print("context creation failed: " + str(e))

    result = triton_client.infer(
                    model_name=model_name,model_version=model_version, inputs=inputs, outputs=outputs)
    triton_client.close()

    


model_names = [ 'xception',"resnet", 'inception']
model_versions = [['1', '2'], ['1', '2', '3'], ['1','2']]
results = []
inputs = []
for bat in [2,4,8]:
    batch =create_batch_image(bat)
    inputs.append(
                    httpclient.InferInput(
                        name="input", shape=batch.shape, datatype="FP32")
                )
    inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)
    
    outputs = []
    outputs.append(httpclient.InferRequestedOutput(name="output"))
    for i in range(100):
        for j,model_name in enumerate(model_names):
            for version in model_versions[j]:
                print(model_name, version)
                threading.Thread(target=send_request, args=(model_name,version,)).start()
 