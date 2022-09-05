import requests

inference_request = {
    "inputs": [
        {
          "name": "args",
          "shape": [1],
          "datatype": "BYTES",
          "data": ["This is terrible!"],
        }
    ]
}

requests.post("http://localhost:8080/v2/models/transformer/infer", json=inference_request).json()