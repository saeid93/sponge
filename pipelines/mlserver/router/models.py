import os
import time
from typing import Dict, List
from mlserver import MLModel
import json
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "router"
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


async def model_infer(request_input: InferenceRequest):
    if not LOGS_ENABLED:
        logger.disabled = True
    try:
        inputs = request_input.outputs[0]
        model_name = inputs.parameters.extended_parameters["next_node"]
        if model_name == 'out':
            return model_name, request_input
        # logger.info(f"second node {model_name} data extracted!")
    except:
        inputs = request_input.inputs[0]
        model_name = inputs.parameters.extended_parameters["next_node"]
    logger.info(f"next_node: {model_name}")
    payload_input = InferenceRequest(inputs=[inputs])
    endpoint = f"{model_name}-{model_name}.default.svc.cluster.local:9500"
    async with grpc.aio.insecure_channel(endpoint) as ch:
        output = await send_requests(ch, model_name, payload_input)
    inference_response = converters.ModelInferResponseConverter.to_types(output)
    return model_name, inference_response


class Router(MLModel):
    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        logger.info("Router loaded")
        mlserver.register(
            name="input_requests", description="Measuring number of input requests"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        mlserver.log(input_requests=1)

        # injecting router arrival time to the message
        arrival_time = time.time()
        self.request_counter += 1

        drop_limit_exceed_payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="drop-limit-violation",
                    shape=[1],
                    datatype="BYTES",
                    data=[],
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="text"),
        )

        output = payload
        

        model_name = ''
        while model_name != 'out':
            model_name, output = await model_infer(request_input=output)
        if output.outputs[0].name == "drop-limit-violation":
            return output
        time_so_far = time.time() - arrival_time
        # TODO add the logic of to drop here
        if time_so_far >= DROP_LIMIT:
            drop_message = f"early exit, drop limit exceeded after {model_name.replace('queue-', '')}".encode(
                "utf8"
            )
            drop_limit_exceed_payload.outputs[0].data = [drop_message]
            return drop_limit_exceed_payload

        serving_time = time.time()
        prev_node_name = output.outputs[0].parameters.extended_parameters['node_name']
        prev_arrival = output.outputs[0].parameters.extended_parameters['arrival']
        prev_serving = output.outputs[0].parameters.extended_parameters['serving']
        extended_parameters = {
            "node_name": prev_node_name + [PREDICTIVE_UNIT_ID],
            "arrival": prev_arrival + [arrival_time],
            "serving": prev_serving + [serving_time]}
        output.outputs[0].parameters.extended_parameters = extended_parameters

        return output
