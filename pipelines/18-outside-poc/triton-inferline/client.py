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
deployment_name = 'yolo'
namespace = "default"
model_name = 'yolo'

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
elif model_name == 'yolo':
    yolo_resize = transforms.Resize((640, 640))
    batch = torch.stack(list(map(lambda a: yolo_resize(transform(a)), list(images.values()))))

URL = f"http://localhost:32000/seldon/{namespace}/{deployment_name}"

def predict(data, model_name):
    payload = {
        "inputs": [
            {
                "name": "input",
                "data": data.tolist(),
                "datatype": "FP32",
                "shape": data.shape,
            }
        ]
    }
    print(data.shape)
    r = requests.post(f"{URL}/v2/models/{model_name}/infer", json=payload)
    predictions = np.array(r.json()["outputs"][0]["data"]).reshape(
        r.json()["outputs"][0]["shape"]
    )
    output = [np.argmax(x) for x in predictions]
    return output

if model_name == 'yolo':
    selected = 19
    batch = batch[selected]
    batch = torch.unsqueeze(batch, dim=0)

triton_seldon_output = predict(batch.numpy(), model_name)

if model_name == 'resnet':
    triton_seldon_classes = np.array(classes)[triton_seldon_output]
    print(triton_seldon_classes)
elif model_name == 'yolo':
    pass
