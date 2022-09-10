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

# inputs = []
# replication = 10

# for i in range(10):
#     inputs.append(input_ins)

# payload = {
#     "inputs": inputs
# }


# endpoint = "http://localhost:32000/seldon/default/custom-mlserver/v2/models/infer"
endpoint = "http://localhost:8080/v2/models/node-1/infer"
# response = requests.post(endpoint, json=payload)

batch_test = 1
payload = {
    "inputs": [input_ins]
}

def send_requests():
    response = requests.post(endpoint, json=payload)
    # print('\n')
    # print('-' * 50)
    pp.pprint(response.json())

for i in range(batch_test):
    send_requests()

# responses = []

# for i in range(batch_test):
#     t = threading.Thread(target=send_requests)
#     t.start()
#     responses.append(t)

# for t in responses:
#     t.join()
