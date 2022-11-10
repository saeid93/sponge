from dataclasses import dataclass
from urllib import response
from barazmoon import MLServerAsync
from datasets import load_dataset
import asyncio
import time

# single node inference
gateway_endpoint = "localhost:32000"
deployment_name = 'mock-one'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# single node inference
# gateway_endpoint = "localhost:8080"
# model = 'mock-one'
# endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

# load data
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
data = ds[0]["audio"]["array"].tolist()

http_method = 'post'
workload = [100]
data_shape = [1, len(data)]
data_type = 'audio'

start_time = time.time()

load_tester = MLServerAsync(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(f'{(time.time() - start_time):2.2}s spent in total')
# print(responses)
