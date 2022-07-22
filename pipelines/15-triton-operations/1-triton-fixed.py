# %% [markdown]
# # Triton Server with Minio but no Seldon
#  [source](https://thenewstack.io/deploy-nvidia-triton-inference-server-with-minio-as-model-store/?fr=)
#  use the former notebook to generate model variants
#  ### single node triton server with load test
#  1. Loading to and from minio workflow (multiple models)
#  2. Getting models from [timm](https://timm.fast.ai/)
#  3. Making the node
#  4. Load and unload models operations
#  5. Monitoring
#  6. Naive load test
#  7. TODO Add language models from Huggingface

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



# %%
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
batch.shape

# %%
model_name = 'resnet50'
model = timm.create_model(model_name, pretrained=True)
model.eval()
torch_output = model(batch)
torch_output = torch.nn.functional.softmax(torch_output, dim=1) * 100
torch_output = torch_output.detach().numpy()
torch_output = torch_output.argmax(axis=1)
torch_class = np.array(classes)[torch_output]
torch_class

# %%
# save the onnx model
import torch.onnx
 
model_variant = 1
model_dir = os.path.join(
    'models',
    model_name,
    str(model_variant))
model_path = os.path.join(model_dir, 'model.onnx')
if 'models' not in os.listdir("./"):
    os.makedirs(model_dir)
# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = torch.randn(1, 3, 224, 224)
# Invoke export
torch.onnx.export(
    model, dummy_input,
    model_path,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  'output' : {0 : 'batch_size'}})

# %%


# %%
# use onnx model
import onnx
import onnxruntime
 
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
 
ort_session = onnxruntime.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=['CPUExecutionProvider'])
onnx_output = ort_session.run(None, {'input': batch.numpy()})
onnx_output = torch.nn.functional.softmax(torch.tensor(onnx_output), dim=1)[0] * 100
onnx_output = onnx_output.detach().numpy()
onnx_output = onnx_output.argmax(axis=1)
onnx_class = np.array(classes)[onnx_output]
onnx_class

# %%
# TODO find out why slightly different
 
assert np.all(onnx_output == torch_output)
print(onnx_class)

# %%
%%writefile models/resnet50/config.pbtxt
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size : 100
input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

# %%
VERSION='22.05'
os.system(f"docker pull nvcr.io/nvidia/tritonserver:{VERSION}-py3")
# add --gpus=<number of gpus> on gpu machines
# add -d to run at background and going to the next cell
os.system("docker run --rm -d -p8000:8000 -p8001:8001 -p8002:8002"
          f" -v {os.getcwd()}/models:/models "
          f"nvcr.io/nvidia/tritonserver:{VERSION}-py3"
          " tritonserver --model-repository=/models")

# %% [markdown]
# ### Python Client Examples
#  
#  [examples](https://github.com/triton-inference-server/client/tree/main/src/python/examples)
#  
#  [grpc](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/__init__.py)
#  
#  [http](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/__init__.py)

# %%
print(    "### Python Client Examples\n",
    "\n",
    "[examples](https://github.com/triton-inference-server/client/tree/main/src/python/examples)\n",
    "\n",
    "[grpc](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/__init__.py)\n",
    "\n",
    "[http](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/__init__.py)")

# %%
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:8000', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))

model_name = "resnet50"
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

# %%
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
 
try:
    triton_client = grpcclient.InferenceServerClient(
        url='localhost:8001', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "resnet50"
 
inputs = []
inputs.append(
    grpcclient.InferInput(name="input", shape=batch.shape, datatype="FP32")
)
inputs[0].set_data_from_numpy(batch.numpy())
outputs = []
outputs.append(grpcclient.InferRequestedOutput(name="output"))

result = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
triton_client.close()
triton_output = result.as_numpy('output')
triton_output = torch.nn.functional.softmax(
    torch.tensor(triton_output), dim=1) * 100
triton_output = triton_output.detach().numpy()
triton_output = triton_output.argmax(axis=1)
triton_class = np.array(classes)[triton_output]
triton_class

# %%
def config_builder(
  name: str, platform: str, max_batch_size: int,
  ):
  config = (f"name: \"{name}\"\n"
            f"platform: \"{platform}\"\n"
            f"max_batch_size: {str(max_batch_size)}")
  common_config="""
input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
version_policy: { all { }}
  """
  return config + common_config
 
print(config_builder('resnet50', 'onnxruntime_onnx', 100))

# %%
from typing import List
 
def generate_model_variants(model_name: str = 'resnet',
    versions: list = ['18', '34', '101']):
    # model name
    timm_models = timm.list_models(model_name+'*', pretrained=True)
    model_path = os.path.join(
        'models',
        model_name,
    )
    config_path = os.path.join(
        model_path,
        'config.pbtxt')
    # if 'models' not in os.listdir("./"):
    os.makedirs(model_path)
    config = config_builder(
        name=model_name,
        platform='onnxruntime_onnx',
        max_batch_size=100)
    with open(config_path, 'w') as f:
        f.write(config)
    for variant_id, model_variant in enumerate(versions):
        model_full_name = model_name + model_variant
        if not model_full_name in timm_models:
            raise ValueError(
                f"Model {model_full_name} does not exist"
            )
        model = timm.create_model(model_full_name, pretrained=True)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        model_variant_dir = os.path.join(model_path, str(variant_id+1))
        model_variant_path = os.path.join(model_variant_dir, 'model.onnx')
        # if 'models' not in os.listdir("./"):
        os.makedirs(model_variant_dir)
        torch.onnx.export(
            model, dummy_input,
            model_variant_path,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={'input' : {0 : 'batch_size'},
                          'output' : {0 : 'batch_size'}})
 
def model_generator(
    model_names: List[str],
    versions: List[List[str]]):
    assert len(model_names) == len(versions),\
        "length modes list {} does not match versions list {}".fromat(
            len(model_names),
            len(versions)
        )
    for model_name, version in zip(model_names, versions):
        generate_model_variants(
            model_name=model_name,
            versions=version
        )
 
# read these from json/yamls build with a proper config builder
model_generator(
    model_names = ['resnet', 'xception'],
    versions = [['18', '34', '101'], ['', '41', '65', '71']]
)



# %%
# generate models
VERSION='22.05'
os.system(f"docker pull nvcr.io/nvidia/tritonserver:{VERSION}-py3")
# add --gpus=<number of gpus> on gpu machines
# add -d to run at background and going to the next cell
os.system("docker run --rm -d -p8000:8000 -p8001:8001 -p8002:8002"
          f" -v {os.getcwd()}/models:/models "
          f"nvcr.io/nvidia/tritonserver:{VERSION}-py3"
          " tritonserver --model-repository=/models")
# print("docker run --rm -d -p8000:8000 -p8001:8001 -p8002:8002"
#       f" -v {os.getcwd()}/models:/models "
#       f"nvcr.io/nvidia/tritonserver:{VERSION}-py3"
#       " tritonserver --strict-model-config=false --model-repository=/models")

# %%
# send request to multi-models
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
 
 
model_version = "1"
 
try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:8000', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "resnet"

inputs = []
inputs.append(
    httpclient.InferInput(name="input", shape=batch[0:4].shape, datatype="FP32")
)
inputs[0].set_data_from_numpy(batch[0:4].numpy(), binary_data=False)
 
outputs = []
outputs.append(httpclient.InferRequestedOutput(name="output"))

result = triton_client.infer(
    model_name=model_name,
    model_version=model_version, # different form the older model
    inputs=inputs, outputs=outputs)
triton_client.close()
# result.get_response()
triton_output = result.as_numpy('output')
triton_output = torch.nn.functional.softmax(
    torch.tensor(triton_output), dim=1) * 100
triton_output = triton_output.detach().numpy()
triton_output = triton_output.argmax(axis=1)
triton_class = np.array(classes)[triton_output]
print(triton_class)

# %%
triton_class

# %%
# generate models
VERSION='22.05'
MODEL_MANAGEMENT = 'explicit'
os.system(f"docker pull nvcr.io/nvidia/tritonserver:{VERSION}-py3")
# add --gpus=<number of gpus> on gpu machines
# add -d to run at background and going to the next cell
os.system("docker run --rm -d -p8000:8000 -p8001:8001 -p8002:8002"
          f" -v {os.getcwd()}/models:/models "
          f"nvcr.io/nvidia/tritonserver:{VERSION}-py3"
          " tritonserver --model-repository=/models"
          f" --model-control-mode={MODEL_MANAGEMENT} --load-model=*")

# %%
# see 
# https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_model_control.py
# https://github.com/triton-inference-server/server/blob/main/docs/model_management.md
# https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md
 
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
 
 
try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:8000', verbose=True,
    )
except Exception as e:
     print("context creation failed: " + str(e))
model_name = "resnet"
 
 
print(20*'-' + 'active models' + 20*'-' + '\n')
print(*triton_client.get_model_repository_index(), sep='\n')
 
print(20*'-' + f'unloading model: {model_name}' + 20*'-' + '\n')
print(triton_client.unload_model(model_name))
 
print(20*'-' + 'active models after unloading' + 20*'-' + '\n')
print(*triton_client.get_model_repository_index(), sep='\n')
 
print(20*'-' + f'load model: {model_name}' + 20*'-' + '\n')
print(triton_client.load_model(model_name))
 
print(20*'-' + 'active models after loading back' + 20*'-' + '\n')
print(*triton_client.get_model_repository_index(), sep='\n')
        

# %%
# copy files
!mc mb minio/minio-seldon -p
!mc cp -r ./models minio/minio-seldon

# %%
%%writefile secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-init-container-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: minio
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: minioadmin
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: minioadmin
  RCLONE_CONFIG_S3_ENDPOINT: http://minio.minio-system.svc.cluster.local:9000

# %%
%%writefile seldon-triton.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: resnet
spec:
  name: default
  predictors:
  - graph:
      implementation: TRITON_SERVER
      logger:
        mode: all
      modelUri: s3://minio-seldon/models
      envSecretRefName: seldon-init-container-secret
      name: resnet # This should have the same name as the model inside
      type: MODEL
    name: default
    replicas: 1
  protocol: kfserving

# %%
!kubectl apply -f secret.yaml -n default
!kubectl apply -f seldon-triton.yaml -n default

# %%
!curl -s http://localhost:32000/seldon/default/resnet/v2/models/resnet | jq

# %%
import json
 # from subprocess import PIPE, Popen, run
import requests
 
import numpy as np
 
 
URL = "http://localhost:32000/seldon/default/resnet"
 
 
def predict(data):
    data = {
        "inputs": [
            {
                "name": "input",
                "data": data.tolist(),
                "datatype": "FP32",
                "shape": data.shape,
            }
        ]
    }
 
    r = requests.post(f"{URL}/v2/models/resnet/infer", json=data)
    predictions = np.array(r.json()["outputs"][0]["data"]).reshape(
        r.json()["outputs"][0]["shape"]
    )
    output = [np.argmax(x) for x in predictions]
    return output
triton_seldon_output = predict(batch.numpy())
triton_seldon_classes = np.array(classes)[triton_seldon_output]
print(triton_seldon_classes)



