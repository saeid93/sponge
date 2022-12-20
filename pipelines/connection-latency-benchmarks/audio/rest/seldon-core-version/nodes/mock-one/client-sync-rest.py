from urllib import response
import requests
from pprint import PrettyPrinter
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
pp = PrettyPrinter(indent=4)
from transformers import pipeline
from datasets import load_dataset
import numpy as np
import json
import base64
from mlserver import types

# single node inference
gateway_endpoint = "localhost:32000"
deployment_name = 'mock-one-base64'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/mock-one/infer"

# single node inference
# gateway_endpoint = "localhost:8080"
# model = 'mock-one'
# endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

batch_test = 1

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

def encode_to_bin(im_arr):
    im_bytes = im_arr.tobytes()
    im_base64 = base64.b64encode(im_bytes)
    input_dict = im_base64.decode()
    return input_dict

# # Serializing json
# json_object = json.dumps(input_data, indent=4)
 
# # Writing to sample.json
# with open("input-sample.json", "w") as outfile:
#     outfile.write(json_object)

def send_requests(endpoint):
    payload = {
        "inputs": [
            {
            "name": "parameters-np",
            "datatype": "BYTES",
            "shape": [1, len(input_data)],
            "data": encode_to_bin(input_data),
            "parameters": {
                "content_type": "np",
                "dtype": "f4"
                }
            }
        ]
    }


    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="echo_request",
                shape=[1],
                datatype="BYTES",
                data=[input_data.tobytes()],
                parameters=types.Parameters(content_type = 'base64', dtype='f4', datashape=str([1, len(input_data)])),
            )
        ]
    )
    payload = inference_request.json()

    response = requests.post(endpoint, json=payload)
    return response

# sync version
results = []
for i in range(batch_test):
    response = send_requests(endpoint)
    results.append(response)

pp.pprint(results[0])
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
