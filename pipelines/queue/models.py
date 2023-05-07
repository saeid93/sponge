import os
from mlserver import MLModel
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
import mlserver.types as types
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver
import logging


# Create a FileHandler object and set its level
log_file_path = "./my_logs.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)


# Add the file handler to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "queue"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )

try:
    MODEL_NAME = os.environ["MODEL_NAME"]
    logger.info(f"MODEL_NAME set to: {MODEL_NAME}")
except KeyError as e:
    # MODEL_NAME = "ddd"
    raise ValueError("No model is assigned to this queue")

try:
    LAST_NODE = bool(os.environ["LAST_NODE"])
    logger.info(f"LAST_NODE set to: {LAST_NODE}")
except KeyError as e:
    LAST_NODE = False
    logger.info(f"LAST_NODE env variable not set, using default value: {LAST_NODE}")



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


class Queue(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        mlserver.register(
            name="batch_size", description="Measuring size of the the queue"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        batch_size = payload.inputs[0].shape[0]
        logger.info(f"batch_size: {batch_size}")
        mlserver.log(size_of_queue=batch_size)

        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")
        try:
            # only image and audio model has this attributes
            if payload.inputs[0].shape[0] == 1:
                # logger.info("here")
                # logger.info(f'datashape: {payload.inputs[0].parameters.datashape}')
                payload.inputs[0].parameters.datashape = str(
                    [payload.inputs[0].parameters.datashape]
                )
                payload.inputs[0].parameters.dtype = str(
                    [payload.inputs[0].parameters.dtype]
                )
            else:
                # logger.info("there")
                # logger.info(f'datashape: {payload.inputs[0].parameters.datashape}')
                payload.inputs[0].parameters.datashape = str(
                    payload.inputs[0].parameters.datashape
                )
                payload.inputs[0].parameters.dtype = str(
                    payload.inputs[0].parameters.dtype
                )
        except AttributeError:
            pass
        try:
            if payload.inputs[0].shape[0] == 1:
                payload.inputs[0].parameters.times = str(
                    [payload.inputs[0].parameters.times]
                )
            else:
                payload.inputs[0].parameters.times = str(
                    payload.inputs[0].parameters.times
                )
        except AttributeError:
            pass

        output = await model_infer(model_name=MODEL_NAME, request_input=payload)

        # TODO refactor!
        if output.outputs[0].shape[0] == 1:
            if LAST_NODE:
                if self._settings.max_batch_size == 1:
                    pass
                else:
                    output.outputs[0].parameters.times = eval(
                        output.outputs[0].parameters.times
                    )
            elif self._settings.max_batch_size == 1:
                output.outputs[0].parameters.times = str(
                    output.outputs[0].parameters.times
                )
            else:
                output.outputs[0].parameters.times = eval(
                    output.outputs[0].parameters.times
                )
        else:
            output.outputs[0].parameters.times = eval(
                output.outputs[0].parameters.times
            )

        # TODO refactor!
        try:
            if output.outputs[0].shape[0] == 1:
                if LAST_NODE:
                    if self._settings.max_batch_size == 1:
                        pass
                    else:
                        output.outputs[0].parameters.datashape = eval(
                            output.outputs[0].parameters.datashape
                        )
                else:
                    output.outputs[0].parameters.datashape = str(
                        output.outputs[0].parameters.datashape
                    )
            else:
                output.outputs[0].parameters.datashape = eval(
                    output.outputs[0].parameters.datashape
                )
        except AttributeError:
            pass

        return output
