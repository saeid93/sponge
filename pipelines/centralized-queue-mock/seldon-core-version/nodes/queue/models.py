import os
import time
from mlserver import MLModel
import json
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
# from typing import List
# import mlserver.types as types
# import grpc
# from mlserver.codecs.string import StringRequestCodec
# from mlserver.codecs.numpy import NumpyRequestCodec
# import mlserver.grpc.dataplane_pb2_grpc as dataplane
# import mlserver.grpc.converters as converters
import mlserver

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "queue"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


class Queue(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        logger.info("Router loaded")
        # mlserver.register(
        #     name="input_requests", description="Measuring number of input requests"
        # )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # mlserver.log(input_requests=1)

        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")

        # TODO endpoint to models?

        logger.info(f"payload: {payload}")
        inference_response = InferenceResponse(
            outputs=payload.inputs, model_name=self.name)
        return inference_response
