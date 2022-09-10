import requests
from pprint import PrettyPrinter
import threading

pp = PrettyPrinter(indent=4)

input_ins = {
    "name": "parameters-np",
    "datatype": "FP32",
    "shape": [2, 2],
    "data": [1, 2, 3, 4],
    "parameters": {
        "content_type": "np"
        }
    }

# endpoint = "http://localhost:8080/v2/models/node-1/infer"
endpoint = "http://localhost:32000/seldon/default/custom-mlserver/v2/models/infer"
# response = requests.post(endpoint, json=payload)

batch_test = 1
payload = {
    "inputs": [input_ins]
}

def send_requests():
    response = requests.post(endpoint, json=payload)
    print(response)
    # print('\n')
    # print('-' * 50)
    # pp.pprint(response.json())

responses = []

for i in range(batch_test):
    response = requests.post(endpoint, json=payload)
    responses.append(response)

a = 1

# for i in range(batch_test):
#     t = threading.Thread(target=send_requests)
#     t.start()
#     responses.append(t)

# for t in responses:
#     t.join()
