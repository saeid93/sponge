from dataclasses import dataclass
from urllib import response
from barazmoon import MLServerAsyncGrpc
from datasets import load_dataset
import asyncio
import json
import time


# load data
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
# data = ds[0]["audio"]["array"].tolist()
data = ds[0]["audio"]["array"].tolist()

load = 1
test_duration = 10
variant = 0
platform = 'seldon'
mode = 'equal'

# single node inference
if platform == 'seldon':
    endpoint = "localhost:32000"
    deployment_name = 'mock-one'
    model = 'mock-one'
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == 'mlserver':
    endpoint = "localhost:8081"
    model = 'mock-one'
    metadata = []

http_method = 'post'
workload = [load] * test_duration
data_shape = [1, len(data)]
data_type = 'audio'

start_time = time.time()

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
    data=data,
    mode=mode, # options - step, equal, exponential
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(f'{(time.time() - start_time):2.2}s spent in total')

import matplotlib.pyplot as plt
import numpy as np

# roundtrip latency
roundtrip_lat = []
for sec_resps in responses:
    for resp in sec_resps:
        times = resp['timing']
        sending_time = times['arrival_time'] - times['sending_time']
        roundtrip_lat.append(sending_time)
fig, ax = plt.subplots()
ax.plot(np.arange(len(roundtrip_lat)), roundtrip_lat)
ax.set(xlabel='request id', ylabel='roundtrip latency (s)', title=f'roundtrip latency, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig(f"{platform}_variant_{variant}-grpc-roundtrip_lat-load-{load}-test_duration-{test_duration}.png")
plt.show()

# sending time
start_times = []
for sec_resps in responses:
    for resp in sec_resps:
        times = resp['timing']
        sending_time = times['sending_time'] - start_time
        start_times.append(sending_time)
fig, ax = plt.subplots()
ax.plot(np.arange(len(start_times)), start_times)
ax.set(xlabel='request id', ylabel='sending time (s)', title=f'load tester sending time, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig(f"{platform}_variant_{variant}-grpc-sending_time-load-{load}-test_duration-{test_duration}.png")
plt.show()

# server arrival time
server_arrival_time = []
for sec_resps in responses:
    for resp in sec_resps:
        server_recieving_time = json.loads(resp['outputs'][0]['data'][0])['time']['arrival_mock_one'] - start_time
        server_arrival_time.append(server_recieving_time)
fig, ax = plt.subplots()
ax.plot(np.arange(len(server_arrival_time)), server_arrival_time)
ax.set(xlabel='request id', ylabel='server arrival time (s)', title=f'Server recieving time of requests, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig(f"{platform}_variant_{variant}-rest-server_arrival_time_from_start-load-{load}-test_duration-{test_duration}.png")
plt.show()

# server arrival latency
server_arrival_latency = []
for sec_resps in responses:
    for resp in sec_resps:
        times = resp['timing']
        server_recieving_time = json.loads(resp['outputs'][0]['data'][0])['time']['arrival_mock_one'] - times['sending_time']
        server_arrival_latency.append(server_recieving_time)
fig, ax = plt.subplots()
ax.plot(np.arange(len(server_arrival_latency)), server_arrival_latency)
ax.set(xlabel='request id', ylabel='server arrival latency (s)', title=f'Server recieving latency, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig(f"{platform}_variant_{variant}-rest-server_recieving_latency-load-{load}-test_duration-{test_duration}.png")
plt.show()