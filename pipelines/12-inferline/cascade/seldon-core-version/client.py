import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image

from seldon_core.seldon_client import SeldonClient


def getImage(path):
    img = image.load_img(path, target_size=(227, 227))
    x = image.img_to_array(img)
    plt.imshow(x / 255.0)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


X = getImage("car.png")
X = X.transpose((0, 3, 1, 2))
print(X.shape)

sc = SeldonClient(deployment_name="openvino-model", namespace="seldon")

response = sc.predict(
    gateway_endpoint="localhost:32000",gateway="istio", transport="grpc", data=X, client_return_type="proto"
)

result = response.response.data.tensor.values

result = np.array(result)
result = result.reshape(1, 1000)

with open("imagenet_classes.json") as f:
    cnames = eval(f.read())

    for i in range(result.shape[0]):
        single_result = result[[i], ...]
        ma = np.argmax(single_result)
        print("\t", i, cnames[ma])
        assert cnames[ma] == "sports car, sport car"