import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

payload = {
    "inputs": [
        {
            "name": "parameters-np",
            "datatype": "FP32",
            "shape": [2, 2],
            "data": [1, 2, 3, 4],
            "parameters": {
                "content_type": "np"
            }
        },
        {
            "name": "parameters-np",
            "datatype": "FP32",
            "shape": [2, 2],
            "data": [1, 2, 3, 4],
            "parameters": {
                "content_type": "np"
            }
        }
    ]
}


# endpoint = "http://localhost:32000/seldon/default/custom-mlserver/v2/models/infer"
endpoint = "http://localhost:8080/v2/models/node-1/infer"
response = requests.post(endpoint, json=payload)

pp.pprint(response.json())