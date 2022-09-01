import requests

x_0 = [28.0]
inference_request = {
    "inputs": [
        {
          "name": "marriage",
          "shape": [1],
          "datatype": "FP32",
          "data": x_0
        }
    ]
}

endpoint = "http://localhost:32000/seldon/default/models/v2/models/blank-model/infer"
response = requests.post(endpoint, json=inference_request)

response.json()