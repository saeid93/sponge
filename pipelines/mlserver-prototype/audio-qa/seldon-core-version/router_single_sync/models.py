import os
import time
from mlserver import MLModel
# import torch
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse)
from mlserver import MLModel
from typing import List
import mlserver.types as types
import grpc
from mlserver.codecs.string import StringRequestCodec
from mlserver.codecs.numpy import NumpyRequestCodec
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters


try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.info(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'audio'
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")


# async def send_requests(ch, model_name, payload, metadata):
#     grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

#     inference_request_g = converters.ModelInferRequestConverter.from_types(
#         payload, model_name=model_name, model_version=None
#     )
#     response = await grpc_stub.ModelInfer(
#         request=inference_request_g,
#         metadata=metadata)
#     return response

def send_requests(ch, model_name, payload, metadata):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = grpc_stub.ModelInfer(
        request=inference_request_g,
        metadata=metadata)
    return response

class Router(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        logger.info('Router loaded')
        self.loaded = True
        return self.loaded


    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        t0 = time.time()
        arrival_time = time.time()
        request_input = payload.inputs[0]
        self.request_counter += 1

        endpoint = "localhost:32000"
        namespace = "default"
        logger.info(f"request counter:\n{self.request_counter}\n")

        # --------- model one ---------
        deployment_name_one = 'audio'
        model_name_one = 'audio'
        
        metadata_one = [("seldon", deployment_name_one), ("namespace", namespace)]
        print(f"transform zero time: {time.time() - t0}")
        t1 = time.time()
        payload_input = types.InferenceRequest(
            inputs=[request_input]
        )
        print(f"trandform one time: {time.time() - t1}")
        t2 = time.time()
        # async with grpc.aio.insecure_channel(endpoint) as ch:
        #     output_one = await send_requests(ch, model_name_one, payload_input, metadata_one)
        ch = grpc.insecure_channel(endpoint)
        output_one = send_requests(ch, model_name_one, payload_input, metadata_one)

        print(f"trandform two time: {time.time() - t2}")
        t3 = time.time()
        inference_response_one = \
            converters.ModelInferResponseConverter.to_types(output_one)
        print(f"trandform two time: {time.time() - t3}")
        return inference_response_one