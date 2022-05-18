# %%
import torchvision
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os

data_folder_path = '/home/cc/inference/infernece-pipeline-joint-optimization/pipelines/12-inferline/cascade/data'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)
def show(img):
    img_2 = Image.open(os.path.join(dataset_folder_path, img))
    img_2.show()

def load_pics(img):
    img = Image.open(os.path.join(dataset_folder_path, img))
    return img

def filter_color_images(img):
    img_2 = Image.open(os.path.join(dataset_folder_path, img[0]))
    if img_2.mode == 'RGB':
        return True
    return False


# %%
with open(classes_file_path) as f:
    classes = [line.strip() for line in f.readlines()]

x = np.array([])
directory = os.fsencode(dataset_folder_path)

for root, dirs, files in os.walk(dataset_folder_path):
    for filename in files:
        x = np.append(x, filename)
df = pd.DataFrame(data=x, columns=["images"])
df['images'][0]

# %%
# client side preprocess
df = df.sort_values(by=['images'])

df_s = df.head(5)
df_s = df_s[df_s.apply(filter_color_images, axis=1)]
df_s['images'] = df_s['images'].apply(load_pics)

type(df_s['images'].iloc[0])

# %%
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image

from seldon_core.seldon_client import SeldonClient


# def getImage(path):
#     img = image.load_img(path, target_size=(227, 227))
#     x = image.img_to_array(img)
#     plt.imshow(x / 255.0)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x


# X = getImage("car.png")
# X = X.transpose((0, 3, 1, 2))
# print(X.shape)


# single node inferline

gateway_endpoint="localhost:32000"
deployment_name = 'inferline-cascade'
namespace = "default"

X = np.array(df_s['images'].iloc[1])

sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

response = sc.predict(
    data=X
)

# result = np.array(response.response['data']['tensor']['values']).astype(np.uint8).reshape(X.shape)

print(f"resnet indicies: {response.response['jsonData']['indices']}")
print(f"resnet max_prob_percentage: {response.response['jsonData']['max_prob_percentage']}")
print(f"resnet percentages: {response.response['jsonData']['percentages']}")
# single node resnet






# pipeline
