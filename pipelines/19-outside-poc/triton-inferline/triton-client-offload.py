import timm
import torch
from PIL import Image
import os
import numpy as np
import pandas as pd

# os.system('sudo umount -l ~/my_mounting_point')
# os.system('cc-cloudfuse mount ~/my_mounting_point')
 
# data_folder_path = '/home/cc/my_mounting_point/datasets'
# dataset_folder_path = os.path.join(
#     data_folder_path, 'ILSVRC/Data/DET/test'
# )
# classes_file_path = os.path.join(
#     data_folder_path, 'imagenet_classes.txt'
# )
 
# image_names = os.listdir(dataset_folder_path)
# image_names.sort()
# with open(classes_file_path) as f:
#     classes = [line.strip() for line in f.readlines()]

# def image_loader(folder_path, image_name):
#     image = Image.open(
#         os.path.join(folder_path, image_name))
#     # if there was a need to filter out only color images
#     # if image.mode == 'RGB':
#     #     pass
#     return image
# num_loaded_images = 4
# images = {
#     image_name: image_loader(
#         dataset_folder_path, image_name) for image_name in image_names[
#             :num_loaded_images]}


# from torchvision import transforms
 
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )])
 
# batch = torch.stack(list(map(lambda a: transform(a), list(images.values()))))


# def config_builder(
#   name: str, platform: str, max_batch_size: int,
#   ):
#   config = (f"name: \"{name}\"\n"
#             f"platform: \"{platform}\"\n"
#             f"max_batch_size: {str(max_batch_size)}")
#   common_config="""
# input [
#   {
#     name: "input"
#     data_type: TYPE_FP32
#     format: FORMAT_NCHW
#     dims: [ 3, 224, 224 ]
#   }
# ]
# output [
#   {
#     name: "output"
#     data_type: TYPE_FP32
#     dims: [ 1000 ]
#   }
# ]
# version_policy: { all { }}
#   """
#   return config + common_config
 
# print(config_builder('resnet50', 'onnxruntime_onnx', 100))

# from typing import List
 
# def generate_model_variants(model_name: str = 'resnet',
#     versions: list = ['18', '34', '101']):
#     # model name
#     timm_models = timm.list_models(model_name+'*', pretrained=True)
#     model_path = os.path.join(
#         'models',
#         model_name,
#     )
#     config_path = os.path.join(
#         model_path,
#         'config.pbtxt')
#     # if 'models' not in os.listdir("./"):
#     os.makedirs(model_path)
#     config = config_builder(
#         name=model_name,
#         platform='onnxruntime_onnx',
#         max_batch_size=100)
#     with open(config_path, 'w') as f:
#         f.write(config)
#     for variant_id, model_variant in enumerate(versions):
#         model_full_name = model_name + model_variant
#         if not model_full_name in timm_models:
#             raise ValueError(
#                 f"Model {model_full_name} does not exist"
#             )
#         model = timm.create_model(model_full_name, pretrained=True)
#         model.eval()
#         dummy_input = torch.randn(1, 3, 224, 224)
#         model_variant_dir = os.path.join(model_path, str(variant_id+1))
#         model_variant_path = os.path.join(model_variant_dir, 'model.onnx')
#         # if 'models' not in os.listdir("./"):
#         os.makedirs(model_variant_dir)
#         torch.onnx.export(
#             model, dummy_input,
#             model_variant_path,
#             input_names = ['input'],
#             output_names = ['output'],
#             dynamic_axes={'input' : {0 : 'batch_size'},
#                           'output' : {0 : 'batch_size'}})
 
# def model_generator(
#     model_names: List[str],
#     versions: List[List[str]]):
#     assert len(model_names) == len(versions),\
#         "length modes list {} does not match versions list {}".fromat(
#             len(model_names),
#             len(versions)
#         )
#     for model_name, version in zip(model_names, versions):
#         generate_model_variants(
#             model_name=model_name,
#             versions=version
#         )
 
# # read these from json/yamls build with a proper config builder
# model_generator(
#     model_names = ['resnet', 'xception'],
#     versions = [['18', '34', '101'], ['', '41', '65', '71']]
# )

 
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


deployment_name = 'triton-nostorage'
namespace = "default"

URL = f"localhost:32000/seldon/{namespace}/{deployment_name}"
try:
    triton_client = httpclient.InferenceServerClient(
        url=URL, verbose=False,
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
