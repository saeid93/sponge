from pprint import PrettyPrinter
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
from datasets import load_dataset
import mlserver.types as types
import grpc
import json

pp = PrettyPrinter(indent=4)

# single node inference
# endpoint = "localhost:32000"
# deployment_name = 'audio-qa'
# model = 'audio-qa'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node inference
endpoint = "localhost:8081"
model = 'audio-qa'
metadata = []
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

batch_test = 100
responses = []

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

def send_requests():
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="echo_request",
                shape=[1, len(input_data)],
                datatype="FP32",
                data=input_data.tolist(),
                parameters=types.Parameters(content_type="np"),
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