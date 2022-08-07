import os
import PIL
from PIL import Image
from typing import Dict
import requests
import torch
from torchvision import transforms
import numpy as np


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
num_loaded_images = 30

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

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'resnet'
namespace = "default"
model_name = 'resnet'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])
 
if model_name == 'resnet':
    batch = torch.stack(list(map(lambda a: transform(a), list(images.values()))))

URL = f"localhost:32000/seldon/{namespace}/{deployment_name}"

# ------ Infer without model version ------

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

try:
    triton_client = httpclient.InferenceServerClient(
        url=URL, verbose=False
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
print(triton_class)


# ------ Infer with model version ------


# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'resnet'
namespace = "default"
model_name = 'resnet'
model_version = '1'

URL = f"localhost:32000/seldon/{namespace}/{deployment_name}"

try:
    triton_client = httpclient.InferenceServerClient(
        url=URL, verbose=False
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "resnet"

inputs = []
inputs.append(
    httpclient.InferInput(name="input", shape=batch.shape, datatype="FP32")
)
inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)
 
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


# ------ Infer with GRPC ------

# import tritonclient.grpc as grpcclient
# from tritonclient.utils import InferenceServerException
 
# try:
#     triton_client = grpcclient.InferenceServerClient(
#         url=URL, verbose=False
#     )
# except Exception as e:
#     print("context creation failed: " + str(e))
# model_name = "resnet50"
 
# inputs = []
# inputs.append(
#     grpcclient.InferInput(name="input", shape=batch.shape, datatype="FP32")
# )
# inputs[0].set_data_from_numpy(batch.numpy())
# outputs = []
# outputs.append(grpcclient.InferRequestedOutput(name="output"))

# result = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
# triton_client.close()
# triton_output = result.as_numpy('output')
# triton_output = torch.nn.functional.softmax(
#     torch.tensor(triton_output), dim=1) * 100
# triton_output = triton_output.detach().numpy()
# triton_output = triton_output.argmax(axis=1)
# triton_class = np.array(classes)[triton_output]
# print(triton_class)


import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'resnet'
namespace = "default"
model_name = 'resnet'
model_version = '1'

URL = f"localhost:32000/seldon/{namespace}/{deployment_name}"

try:
    triton_client = grpcclient.InferenceServerClient(
        url=URL, verbose=False
    )
except Exception as e:
    print("context creation failed: " + str(e))
model_name = "resnet"

inputs = []
inputs.append(
    grpcclient.InferInput(name="input", shape=batch.shape, datatype="FP32")
)
inputs[0].set_data_from_numpy(batch.numpy())

outputs = []
outputs.append(grpcclient.InferRequestedOutput(name="output"))

result = triton_client.infer(
    model_name=model_name,
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