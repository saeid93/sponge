import requests
from pprint import PrettyPrinter
import threading
import numpy as np
from copy import deepcopy
pp = PrettyPrinter(indent=4)

batch_test = 23

input_ins = {
    "name": "parameters-np",
    "datatype": "FP32",
    "shape": [1,2],
    "data": [[12, 43]],
    # "parameters": {
    #     "content_type": "np"
    #     }
    }


payloads = []

base_data = np.zeros(2)
for i in range(batch_test):
    base_data += 1
    input_ins['data'] = deepcopy(base_data).tolist()
    payload = {
    "inputs": [deepcopy(input_ins)]
    }
    payloads.append(payload)




# endpoint = "http://localhost:32000/seldon/default/custom-mlserver-node-one/v2/models/infer"
endpoint = "http://localhost:8080/v2/models/node-1/infer"
# response = requests.post(endpoint, json=payload)

batch_test = 6


responses = []
def send_requests(index):
    response = requests.post(endpoint, json=payloads[index])
    # print('\n')
    # print('-' * 50)
    # pp.pprint(response.json())
    responses.append(response)
    return response

# for i in range(batch_test):
#     send_requests()

thread_pool = []

data_to_send = map(
    lambda l: l.tolist(),
    [1,2])

data = np.zeros(2)
for i in range(batch_test):
    t = threading.Thread(target=send_requests, args=[i])
    t.start()
    thread_pool.append(t)

for t in thread_pool:
    t.join()


pp.pprint(list(map(lambda l:l.json(), responses)))