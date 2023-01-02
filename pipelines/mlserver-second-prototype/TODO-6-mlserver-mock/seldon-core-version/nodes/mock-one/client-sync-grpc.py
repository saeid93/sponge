from pprint import PrettyPrinter
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
from datasets import load_dataset
import mlserver.types as types
import json
import grpc
# from mlserver_huggingface.codecs import C

pp = PrettyPrinter(indent=4)


# single node inference
endpoint = "localhost:32000"
deployment_name = 'mock-one'
model = 'mock-one'
namespace = "default"
metadata = [("seldon", deployment_name), ("namespace", namespace)]
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node inference
# endpoint = "localhost:8081"
# model = 'mock-one'
# metadata = []
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

batch_test = 1

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]
import numpy as np
def send_requests():
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="echo_request",
                shape=[1, len(input_data)],
                datatype="BYTES",
                data=input_data.tobytes(),
                parameters=types.Parameters(content_type="base64"),
            )
        ]
    )
    inference_request_g = converters.ModelInferRequestConverter.from_types(
        inference_request, model_name=model, model_version=None
    )
    response = grpc_stub.ModelInfer(
        request=inference_request_g,
        metadata=metadata)
    return response


# sync version
results = []
for i in range(batch_test):
    response = send_requests()
    results.append(response)


# Note that here we just convert from the gRPC types to the MLServer types
inference_response = ModelInferResponseConverter.to_types(response)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
