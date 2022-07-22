# %% [markdown]
# # Triton Server with Minio but no Seldon
# [source](https://thenewstack.io/deploy-nvidia-triton-inference-server-with-minio-as-model-store/?fr=)
# use the former notebook to generate model variants

# %%
!kubectl create secret generic aws-credentials --from-literal=AWS_ACCESS_KEY_ID=minioadmin --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin

# %%
%%writefile triton-deploy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: triton
  name: triton
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - image: nvcr.io/nvidia/tritonserver:21.09-py3
        name: tritonserver
        command: ["/bin/bash"]
        args: ["-c", "cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates && update-ca-certificates && /opt/tritonserver/bin/tritonserver --model-store=s3://http://minio.minio-system.svc.cluster.local:9000/minio-seldon/models --strict-model-config=false"]
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: AWS_SECRET_ACCESS_KEY      
        ports:
          - containerPort: 8000
            name: http
          - containerPort: 8001
            name: grpc
          - containerPort: 8002
            name: metrics
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory

# %%
%%writefile triton-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: triton
spec:
  type: NodePort
  selector:
    app: triton
  ports:
    - protocol: TCP
      name: http
      port: 8000
      nodePort: 30800
      targetPort: 8000
    - protocol: TCP
      name: grpc
      port: 8001
      nodePort: 30801
      targetPort: 8001
    - protocol: TCP
      name: metrics
      nodePort: 30802
      port: 8002
      targetPort: 8002

# %%
!kubectl apply -f triton-deploy.yaml
!kubectl apply -f triton-service.yaml

# %%
import timm
import torch
from PIL import Image
import os
import numpy as np
import pandas as pd

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
num_loaded_images = 4
images = {
    image_name: image_loader(
        dataset_folder_path, image_name) for image_name in image_names[
            :num_loaded_images]}
# load and transform model
from torchvision import transforms
 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
 
batch = torch.stack(list(map(lambda a: transform(a), list(images.values()))))

# %%
batch

# %%
import onnx
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:30800', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "resnet"
 
inputs = []
inputs.append(
    httpclient.InferInput(
        name="input", shape=batch.shape, datatype="FP32")
)
inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)
 
outputs = []
outputs.append(httpclient.InferRequestedOutput(name="output"))
 
result = triton_client.infer(
    model_name=model_name, inputs=inputs, outputs=outputs)
triton_client.close()
triton_output = result.as_numpy('output')
triton_output = torch.nn.functional.softmax(
    torch.tensor(triton_output), dim=1) * 100
triton_output = triton_output.detach().numpy()
triton_output = triton_output.argmax(axis=1)
triton_class = np.array(classes)[triton_output]
triton_class
 
 # stop triton server

# %%
print(    "import onnx\n",
    "    triton_client = httpclient.InferenceServerClient(\n",
    "        url='localhost:30800', verbose=True\n",
    "    )\n",
    "except Exception as e:\n",
    "    print(\"context creation failed: \" + str(e))\n",
    "model_name = \"resnet\"\n",
    "\n",
    "inputs = []\n",
    "inputs.append(\n",
    "    httpclient.InferInput(\n",
    "        name=\"input\", shape=batch.shape, datatype=\"FP32\")\n",
    ")\n",
    "inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)\n",
    "\n",
    "outputs = []\n",
    "outputs.append(httpclient.InferRequestedOutput(name=\"output\"))\n",
    "\n",
    "result = triton_client.infer(\n",
    "    model_name=model_name, inputs=inputs, outputs=outputs)\n",
    "triton_client.close()\n",
    "triton_output = result.as_numpy('output')\n",
    "triton_output = torch.nn.functional.softmax(\n",
    "    torch.tensor(triton_output), dim=1) * 100\n",
    "triton_output = triton_output.detach().numpy()\n",
    "triton_output = triton_output.argmax(axis=1)\n",
    "triton_class = np.array(classes)[triton_output]\n",
    "triton_class\n",
    "\n",
    "# stop triton server")

# %%



