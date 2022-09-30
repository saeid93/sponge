import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec

# # single node inferline
# gateway_endpoint="localhost:32000"
# deployment_name = 'nlp-sum'
# namespace = "default"

# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# single node inferline
gateway_endpoint="localhost:8080"
model='nlp-qa'
endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

def send_requests(endpoint, data):
    payload = {
        "inputs": [
            {
            "name": "text_inputs",
            "shape": [1],
            "datatype": "BYTES",
            "data": data,
            "parameters": {
                "content_type": "str"
            }
            }
        ]
    }
    response = requests.post(endpoint, json=payload)
    return response

data=['{"time": {"arrival_nlp_sum": 1664473950.8936176, "serving_nlp_sum":'
      ' 1664473952.8370514}, "output": {"summary_text": " Yoga is an antidote'
      ' for stress and a pathway for deeper understanding of yourself and'
      ' others . It combines the elements of yoga with elements of balance'
      ' and focus . The classes are just plain wonderful: they are just a'
      ' few moments away from the demands of life\'s demands . He is an RYT'
      ' 500 certified yoga teacher and author of YogaWorks ."}}']


# sync version
results = []
for data_ins in data:
    response = send_requests(endpoint, data_ins)
    results.append(response)

pp.pprint(results[0])
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
