import os
import pathlib
from PIL import Image
import numpy as np
from mlserver.types import InferenceResponse
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
import requests
from pprint import PrettyPrinter
import mlserver.types as types
import grpc
import json

pp = PrettyPrinter(indent=4)


def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    return image

# single node mlserver
endpoint = "localhost:8081"
model = 'yolo'
metadata = []
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = 'yolo'
# model = 'yolo'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)


PATH = pathlib.Path(__file__).parent.resolve()
input_data = image_loader(PATH, 'input-sample.JPEG')
with open(os.path.join(
    PATH, 'input-sample-shape.json'), 'r') as openfile:
    input_data_shape = json.load(openfile)
    input_data_shape = input_data_shape['data_shape']
images = {}
images['inpue-sample'] = input_data

def send_requests(image):
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="parameters-np",
                shape=[1] + list(np.shape(image)),
                datatype="INT32",
                data=np.array(image).flatten(),
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


# sync version
results = {}
for image_name, image in images.items():
    response = send_requests(image)
    inference_response = ModelInferResponseConverter.to_types(response)
    raw_json = StringRequestCodec.decode_response(inference_response)
    output = json.loads(raw_json[0])
    results[image_name] = output

for image_name, output in results.items():
    print("-"*50, f' {image_name} ', "-"*50)
    output = list(map(lambda l: np.array(l), output['output']['person']))
    pp.pprint(len(output))
