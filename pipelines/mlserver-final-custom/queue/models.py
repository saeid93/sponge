import os
from mlserver import MLModel
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver
import time
from typing import Dict, List


try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "queue"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )

try:
    DROP_LIMIT = float(os.environ["DROP_LIMIT"])
    logger.info(f"DROP_LIMIT set to: {DROP_LIMIT}")
except KeyError as e:
    DROP_LIMIT = 1000
    logger.info(f"DROP_LIMIT env variable not set, using default value: {DROP_LIMIT}")

try:
    MODEL_NAME = os.environ["MODEL_NAME"]
    logger.info(f"MODEL_NAME set to: {MODEL_NAME}")
except KeyError as e:
    raise ValueError("No model is assigned to this queue")

try:
    LAST_NODE = bool(os.environ["LAST_NODE"])
    logger.info(f"LAST_NODE set to: {LAST_NODE}")
except KeyError as e:
    LAST_NODE = False
    logger.info(f"LAST_NODE env variable not set, using default value: {LAST_NODE}")

try:
    LOGS_ENABLED = os.getenv("LOGS_ENABLED", "True").lower() in ("true", "1", "t")
    logger.info(f"LOGS_ENABLED set to: {LOGS_ENABLED}")
except KeyError as e:
    LOGS_ENABLED = True
    logger.info(
        f"LOGS_ENABLED env variable not set, using default value: {LOGS_ENABLED}"
    )

if not LOGS_ENABLED:
    logger.disabled = True


async def send_requests(ch, model_name, payload: InferenceRequest):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(request=inference_request_g, metadata=[])
    return response


async def model_infer(model_name, request_input: InferenceRequest) -> InferenceResponse:
    try:
        inputs = request_input.outputs[0]
    except:
        inputs = request_input.inputs[0]
    payload_input = InferenceRequest(inputs=[inputs])
    endpoint = f"{model_name}-{model_name}.default.svc.cluster.local:9500"
    async with grpc.aio.insecure_channel(endpoint) as ch:
        output = await send_requests(ch, model_name, payload_input)
    inference_response = converters.ModelInferResponseConverter.to_types(output)
    return inference_response


class Queue(MLModel):
    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        mlserver.register(
            name="batch_size", description="Measuring size of the the queue"
        )
        logger.info(f"max_batch_size: {self.settings.max_batch_size}")
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        batch_shape = payload.inputs[0].shape[0]
        logger.info(f"batch_size: {batch_shape}")
        mlserver.log(batch_size=batch_shape)
        self.request_counter += 1

        if type(payload.inputs[0].parameters.extended_parameters) == list:
            payload.inputs[0].parameters.extended_parameters_repeated = payload.inputs[0].parameters.extended_parameters
            payload.inputs[0].parameters.extended_parameters = None

        output = await model_infer(model_name=MODEL_NAME, request_input=payload)

        if type(output.outputs[0].parameters.extended_parameters_repeated) == list:
            output.outputs[0].parameters.extended_parameters = output.outputs[0].parameters.extended_parameters_repeated
            output.outputs[0].parameters.extended_parameters_repeated = None

        return output
