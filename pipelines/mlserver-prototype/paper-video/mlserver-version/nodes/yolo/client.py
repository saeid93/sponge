import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
import requests
from pprint import PrettyPrinter
import threading

PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/sample-image/"
dataset_folder_path = PIPELINES_MODELS_PATH

# os.system('sudo umount -l ~/my_mounting_point')
# os.system('cc-cloudfuse mount ~/my_mounting_point')

# data_folder_path = '/home/cc/my_mounting_point/datasets'
# dataset_folder_path = os.path.join(
#     data_folder_path, 'ILSVRC/Data/DET/test'
# )
# classes_file_path = os.path.join(
#     data_folder_path, 'imagenet_classes.txt'
# )
# with open(classes_file_path) as f:
#     classes = [line.strip() for line in f.readlines()]

image_names = os.listdir(dataset_folder_path)
image_names.sort()

num_loaded_images = 1

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
# gateway_endpoint="localhost:32000"
# deployment_name = 'video-monitoring'
# namespace = "default"

# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

gateway_endpoint="localhost:8080"
endpoint = f"http://{gateway_endpoint}/v2/models/video-1/infer"

def send_requests(endpoint, image):
    input_ins = {
        "name": "parameters-np",
        "datatype": "FP32",
        "shape": list(np.shape(image)),
        "data": np.array(image).tolist(),
        "parameters": {
            "content_type": "np"
            }
        }
    payload = {
    "inputs": [input_ins]
    }
    response = requests.post(endpoint, json=payload)
    return response

# sync version
results = {}
for image_name, image in images.items():
    response = send_requests(endpoint, image)
    results[image_name] = response


# async version
# TODO

# send_requests()
# responses = []

# batch_test = 30000

# responses = []
# def send_requests():
#     response = requests.post(endpoint, json=payload)
#     # print('\n')
#     # print('-' * 50)
#     # pp.pprint(response.json())
#     responses.append(response)
#     return response

# # for i in range(batch_test):
# #     send_requests()

# thread_pool = []

# for i in range(batch_test):
#     t = threading.Thread(target=send_requests)
#     t.start()
#     thread_pool.append(t)

# for t in thread_pool:
#     t.join()


# pp.pprint(list(map(lambda l:l.json(), responses)))
