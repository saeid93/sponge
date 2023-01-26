import os
import time
from mlserver import MLModel
import torch
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver import MLModel
from transformers import pipeline
from typing import List, Dict

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.info(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'nlp-li'
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

try:
    USE_THREADING = bool(os.environ['USE_THREADING'])
    logger.info(f'USE_THREADING set to: {USE_THREADING}')
except KeyError as e:
    USE_THREADING = False
    logger.info(
        f"USE_THREADING env variable not set, using default value: {USE_THREADING}")

try:
    NUM_INTEROP_THREADS = int(os.environ['NUM_INTEROP_THREADS'])
    logger.info(f'NUM_INTEROP_THREADS set to: {NUM_INTEROP_THREADS}')
except KeyError as e:
    NUM_INTEROP_THREADS = 1
    logger.info(
        f"NUM_INTEROP_THREADS env variable not set, using default value: {NUM_INTEROP_THREADS}")

try:
    NUM_THREADS = int(os.environ['NUM_THREADS'])
    logger.info(f'NUM_THREADS set to: {NUM_THREADS}')
except KeyError as e:
    NUM_THREADS = 1
    logger.info(
        f"NUM_THREADS env variable not set, using default value: {NUM_THREADS}")

if USE_THREADING:
    torch.set_num_interop_threads(NUM_INTEROP_THREADS)
    torch.set_num_threads(NUM_THREADS)

class GeneralNLP(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'dinalzein/xlm-roberta-base-finetuned-language-identification'
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.info(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'text-classification' 
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        logger.info('Loading the ML models')
        # TODO add batching like the runtime
        logger.info(f'max_batch_size: {self._settings.max_batch_size}')
        logger.info(f'max_batch_time: {self._settings.max_batch_time}')
        self.model  = pipeline(
            task=self.TASK,
            model=self.MODEL_VARIANT,
            batch_size=self._settings.max_batch_size)
        self.loaded = True
        logger.info('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.info('request input:\n')
            logger.info(f"{request_input}\n")
            decoded_input = self.decode(request_input)
            logger.info(decoded_input)
            X = list(decoded_input)
            batch_shape = request_input.shape[0]
        logger.info(f"recieved batch len:\n{batch_shape}")
        self.request_counter += batch_shape
        self.batch_counter += 1
        logger.info(f"to the model:\n{X}")
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")

        # model part
        output: List[Dict] = self.model(X)
        logger.info(f"model output:\n{output}")
        serving_time = time.time()
        # HACK
        # we have only one language and one upstream node
        # therefore we just directly send the french text
        # to the next node
        output = X

        # times processing
        serving_time = time.time()
        times = {
            PREDICTIVE_UNIT_ID: {
            "arrival": arrival_time,
            "serving": serving_time
            }
        }
        batch_times = [str(times)] * batch_shape
        if self.settings.max_batch_size == 1:
            batch_times = str(batch_times)
        logger.info(f'batch shapes:\n{batch_shape}')
        logger.info(f"batch_times:\n{batch_times}")

        # processing inference response
        output_data = list(map(lambda l: l.encode('utf8'), output))
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="text",
                    shape=[batch_shape],
                    datatype="BYTES",
                    data=output_data,
                    parameters=Parameters(
                        times=batch_times
                    ),
            )],
            model_name=self.name,
            parameters=Parameters(
                type_of='text'
            )
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload