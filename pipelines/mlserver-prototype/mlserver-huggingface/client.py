import requests


inference_request = {
    "inputs": [
        {
          "name": "args",
          "shape": [1],
          "datatype": "BYTES",
          "data": ["I am from Iran, Iran is a shit hole"],
        }
    ]
}

response = requests.post("http://localhost:8080/v2/models/transformer/infer", json=inference_request).json()
print(response)
