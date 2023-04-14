import os
import time
from mlserver import MLModel
import json
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
    PREDICTIVE_UNIT_ID = "router"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


try:
    MODEL_LISTS = json.loads(os.environ["MODEL_LISTS"])
    logger.info(f"MODEL_LISTS set to: {MODEL_LISTS}")
except KeyError as e:
    raise ValueError(f"MODEL_LISTS env variable not set!")


async def send_requests(ch, model_name, payload):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(request=inference_request_g, metadata=[])
    return response


async def model_infer(model_name, request_input):
    try:
        inputs = request_input.outputs[0]
        logger.info(f"second node {model_name} data extracted!")
    except:
        inputs = request_input.inputs[0]
        logger.info(f"first node {model_name} data extracted!")
    payload_input = types.InferenceRequest(inputs=[inputs])
    endpoint = f"{model_name}-{model_name}.default.svc.cluster.local:9500"
    async with grpc.aio.insecure_channel(endpoint) as ch:
        output = await send_requests(ch, model_name, payload_input)
    inference_response = converters.ModelInferResponseConverter.to_types(output)
    return inference_response


class Router(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        logger.info("Router loaded")
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")

        output = payload
        for model_name in MODEL_LISTS:
            logger.info(f"Getting inference responses {model_name}")
            output = await model_infer(model_name=model_name, request_input=output)
        return output
