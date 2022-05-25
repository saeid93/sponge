import os
import PIL
from PIL import Image
from typing import Dict

import numpy as np

from seldon_core.seldon_client import SeldonClient

data_folder_path = '/home/cc/object-store/datasets'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
image_names = os.listdir(dataset_folder_path)

num_loaded_images = 10

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
deployment_name = 'inferline-cascade'
namespace = "saeid"
sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

results = {}
for image_name, image in images.items():
    image = np.array(image)
    response = sc.predict(
        data=image
    )
    results[image_name] = response

for image_name, response in results.items():
    print(f"image name: {image_name}")
    max_prob = response.response['jsonData']['max_prob_percentage']
    print(f"resnet max_prob_percentage: {max_prob}")
    # print() TODO print the path taken 
# print(f"resnet indicies: {response.response['jsonData']['indices']}")
# print(f"resnet max_prob_percentage: {response.response['jsonData']['max_prob_percentage']}")
# print(f"resnet percentages: {response.response['jsonData']['percentages']}")
# print() TODO print the path taken 
