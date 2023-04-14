import os
import time
from mlserver import MLModel

# import torch
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
from typing import List
import mlserver.types as types
import grpc
from mlserver.codecs.string import StringRequestCodec
from mlserver.codecs.numpy import NumpyRequestCodec
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters


try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "audio"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


async def send_requests(ch, model_name, payload, metadata):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(
        request=inference_request_g, metadata=metadata
    )
    return response


class Router(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        logger.info("Router loaded")
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        arrival_time = time.time()
        request_input = payload.inputs[0]
        self.request_counter += 1

        endpoint = "localhost:32000"
        namespace = "default"
        logger.info(f"request counter:\n{self.request_counter}\n")

        # --------- model one ---------
        deployment_name_one = "audio"
        model_name_one = "audio"

        metadata_one = [("seldon", deployment_name_one), ("namespace", namespace)]
        payload_input = types.InferenceRequest(inputs=[request_input])
        async with grpc.aio.insecure_channel(endpoint) as ch:
            output_one = await send_requests(
                ch, model_name_one, payload_input, metadata_one
            )
        inference_response_one = converters.ModelInferResponseConverter.to_types(
            output_one
        )

        # --------- model two ---------
        deployment_name_two = "nlp-qa"
        model_name_two = "nlp-qa"
        metadata_two = [("seldon", deployment_name_two), ("namespace", namespace)]
        input_two = inference_response_one.outputs[0]
        payload_two = types.InferenceRequest(inputs=[input_two])
        async with grpc.aio.insecure_channel(endpoint) as ch:
            payload = await send_requests(ch, model_name_two, payload_two, metadata_two)
        inference_response = converters.ModelInferResponseConverter.to_types(payload)

        # logger.info(f"request counter:\n{self.request_counter}\n")
        # logger.info(f"batch counter:\n{self.batch_counter}\n")
        return inference_response
