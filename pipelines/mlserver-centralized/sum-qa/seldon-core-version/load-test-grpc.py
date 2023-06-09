from urllib import response
from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
import asyncio
import time
import pathlib
import os

load = 1
test_duration = 10
variant = 0
platform = "router"
mode = "equal"


PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, "input-sample-short.txt"), "r") as openfile:
    data = openfile.read()

# times = str([str(request['times']['models'])])

data_shape = [1]
# custom_parameters = {'times': str(times)}
data_1 = Data(
    data=data,
    data_shape=data_shape
    # custom_parameters=custom_parameters
)

# single node inference
if platform == "router":
    endpoint = "localhost:32000"
    deployment_name = "router"
    model = "router"
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "seldon":
    endpoint = "localhost:32000"
    deployment_name = "sum-qa"
    model = None
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "mlserver":
    endpoint = "localhost:8081"
    model = "nlp-sum"
    metadata = []

workload = [load] * test_duration
data_shape = [len(data)]
data_type = "text"

start_time = time.time()

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
    data=[data_1],
    mode=mode,  # options - step, equal, exponential
    data_shape=data_shape,
    data_type=data_type,
)

responses = asyncio.run(load_tester.start())

print(f"{(time.time() - start_time):2.2}s spent in total")
print(responses)
