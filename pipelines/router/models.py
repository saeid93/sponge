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
    SLA = float(os.environ["SLA"])
    logger.info(f"SLA set to: {SLA}")
except KeyError as e:
    SLA = 1000
    logger.info(f"SLA env variable not set, using default value: {SLA}")

try:
    MODEL_LISTS: List[str] = json.loads(os.environ["MODEL_LISTS"])
    logger.info(f"MODEL_LISTS set to: {MODEL_LISTS}")
except KeyError as e:
    raise ValueError(f"MODEL_LISTS env variable not set!")


async def send_requests(ch, model_name, payload: InferenceRequest):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(request=inference_request_g, metadata=[])
    return response


async def model_infer(model_name, request_input: InferenceRequest):
    try:
        inputs = request_input.outputs[0]
        # logger.info(f"second node {model_name} data extracted!")
    except:
        inputs = request_input.inputs[0]
        # logger.info(f"first node {model_name} data extracted!")
    payload_input = InferenceRequest(inputs=[inputs])
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
        mlserver.register(
            name="input_requests", description="Measuring number of input requests"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        mlserver.log(input_requests=1)

        # injecting router arrival time to the message
        arrival_time = time.time()
        pipeline_arrival = {"pipeline_arrival": str(arrival_time)}
        existing_paramteres = payload.inputs[0].parameters
        payload.inputs[0].parameters = existing_paramteres.copy(update=pipeline_arrival)
        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")

        sla_exceed_payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="sla_violaion",
                    shape=[1],
                    datatype="BYTES",
                    data=[],
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="text"),
        )

        output = payload
        for node_index, model_name in enumerate(MODEL_LISTS):
            logger.info(f"Getting inference responses {model_name}")
            output = await model_infer(model_name=model_name, request_input=output)
            if output.outputs[0].name == "sla_violaion":
                logger.info(f"previous step:\n{self.decode(output.outputs[0])}")
                # if "early exit" in self.decode(output.outputs[0]):
                logger.info(f"early exiting from before")
                return output
            existing_paramteres = output.outputs[0].parameters
            output.outputs[0].parameters = existing_paramteres.copy(
                update=pipeline_arrival
            )
            time_so_far = time.time() - arrival_time
            logger.info(f"{model_name} time_so_far:\n{time_so_far}")
            # TODO add the logic of to drop here
            if time_so_far >= SLA and node_index + 1 != len(MODEL_LISTS):
                sla_message = f"early exit, sla exceeded after {model_name.replace('queue-', '')}".encode(
                    "utf8"
                )
                logger.info("early exit from here")
                sla_exceed_payload.outputs[0].data = [sla_message]
                return sla_exceed_payload

        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        model_times: Dict = eval(eval(output.outputs[0].parameters.times)[0])
        model_times.update(times)
        output_times = str([str(model_times)])
        output.outputs[0].parameters.times = output_times

        return output
