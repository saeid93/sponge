import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
import pprint
import requests

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'video-pipeline-with-log'
namespace = "default"
text = ("You can deploy a HuggingFace model"
        " by providing parameters to your pipeline.")

payload = {
    "inputs": [
        {
          "name": "args",
          "shape": [1],
          "datatype": "BYTES",
          "data": ["this is a test"],
        }
    ]
}


ret = requests.post(
    f"http://localhost:32000/seldon/default/gpt2-model/v2/models/transformer/infer", json=payload
)


res = ret.json()

print(ret.json()['outputs'][0]['data'][0])
