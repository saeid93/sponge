import json
import requests
import threading
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=1)

model = 'nlp-qa'

gateway_endpoint = "localhost:8080"
endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

# single node inferline
# gateway_endpoint="localhost:32000"
# deployment_name = 'nlp-qa'
# namespace = "default"

# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


batch_test = 11

responses = []

data=['{"time": {"arrival_nlp_sum": 1664473950.8936176, "serving_nlp_sum":'
      ' 1664473952.8370514}, "output": {"summary_text": " Yoga is an antidote'
      ' for stress and a pathway for deeper understanding of yourself and'
      ' others . It combines the elements of yoga with elements of balance'
      ' and focus . The classes are just plain wonderful: they are just a'
      ' few moments away from the demands of life\'s demands . He is an RYT'
      ' 500 certified yoga teacher and author of YogaWorks ."}}']


def send_requests():
    payload = {
        "inputs": [
            {
                "name": "text_inputs",
                "shape": [1],
                "datatype": "BYTES",
                "data": data,
            }
        ]
    }
    response = requests.post(endpoint, json=payload)
    responses.append(response)
    return response

thread_pool = []

for i in range(batch_test):
    t = threading.Thread(target=send_requests)
    t.start()
    thread_pool.append(t)

for t in thread_pool:
    t.join()


inference_responses = list(map(
    lambda response: InferenceResponse.parse_raw(response.text), responses))
raw_jsons = list(map(
    lambda inference_response: StringRequestCodec.decode_response(
        inference_response), inference_responses))
outputs = list(map(
    lambda raw_json: json.loads(raw_json[0]), raw_jsons))

pp.pprint(outputs)