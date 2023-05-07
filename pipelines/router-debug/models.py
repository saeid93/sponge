import os
import time
from typing import Dict
from mlserver import MLModel
import json
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
import mlserver.types as types
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver
import logging

try:
    POD_NAME = os.environ["POD_NAME"]
    logger.info(f"POD_NAME set to: {POD_NAME}")
except KeyError as e:
    POD_NAME = "resnet-human"
    logger.info(f"POD_NAME env variable not set, using default value: {POD_NAME}")

# File number of files in the folder
num_files = int(len(os.listdir("./logs")) / 2)

# Add the file handler to mlserver logs
log_file_path = f"./logs/{num_files}_log_{POD_NAME}.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Add the file handler for error logs
logger_error = logging.getLogger()
error_log_file_path = f"./logs/{num_files}_error_log_{POD_NAME}.log"
error_file_handler = logging.FileHandler(error_log_file_path)
error_file_handler.setLevel(logging.ERROR)
logger_error.addHandler(error_file_handler)

# Add a stream handler for error logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger_error.addHandler(stream_handler)

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
        mlserver.register(
            name="input_requests", description="Measuring number of input requests"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        mlserver.log(input_requests=1)
        # logger.info(f"payload in:\n{payload}")
        arrival_time = time.time()

        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")

        output = payload
        for model_name in MODEL_LISTS:
            # logger.info(f"{model_name} input:\n{output}")
            logger.info(f"Getting inference responses {model_name}")
            output = await model_infer(model_name=model_name, request_input=output)
            # logger.info(f"{model_name} output:\n{output}")

        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        model_times: Dict = eval(eval(output.outputs[0].parameters.times)[0])
        model_times.update(times)
        output_times = str([str(model_times)])
        output.outputs[0].parameters.times = output_times

        return output
