import os
import time
from mlserver import MLModel
import json
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver

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
    raise ValueError("No model is assigned to this queue")


async def send_requests(ch, model_name, payload):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(request=inference_request_g, metadata=[])
    return response


class Queue(MLModel):
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

        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")
        inference_response = InferenceResponse(
            outputs=payload.inputs, model_name=self.name)

        # TODO next model queue size
        # TODO add request to the model here


        return inference_response
