import os
import time
import json
import asyncio
from mlserver import MLModel
import numpy as np
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver import MLModel
from mlserver.codecs import StringCodec
from mlserver_huggingface.common import NumpyEncoder
from typing import List, Dict
from mlserver import types
import time

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.info(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'node-one'
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

try:
    MODEL_SYNC = eval(os.environ['MODEL_SYNC'])
    logger.info(f'MODEL_SYNC set to: {MODEL_SYNC}')
except KeyError as e:
    MODEL_SYNC = True
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {MODEL_SYNC}")

def decode_from_bin(
    inputs: List[bytes], shapes: List[List[int]], dtypes: List[str]) -> List[np.array]:
    batch = []
    for input, shape, dtype in zip(inputs, shapes, dtypes):
        buff = memoryview(input)
        array = np.frombuffer(buff, dtype=dtype).reshape(shape)
        batch.append(array)
    return batch

async def model(input, sleep):
    if MODEL_SYNC:
        time.sleep(sleep)
    else:
        await asyncio.sleep(sleep)
    _ = input
    output = ["node one output"] * len(input)
    return output
 
class NodeOne(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = float(os.environ['MODEL_VARIANT'])
            logger.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 0.01
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        logger.info('Loading the ML models')
        # TODO add batching like the runtime
        logger.info(f'max_batch_size: {self._settings.max_batch_size}')
        logger.info(f'max_batch_time: {self._settings.max_batch_time}')
        self.model  = model
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        for request_input in payload.inputs:
            dtypes = request_input.parameters.dtype
            shapes = request_input.parameters.datashape
            shapes = list(map(lambda l: eval(l), shapes))
            data = request_input.data.__root__
            X = decode_from_bin(inputs=data, shapes=shapes, dtypes=dtypes)
        arrival_time = time.time()
        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.info(f"to the model:\n{type(X)}")
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")
        output: List[Dict] = await self.model(X, self.MODEL_VARIANT)
        logger.info(f"model output:\n{output}")
        serving_time = time.time()
        times = {
            PREDICTIVE_UNIT_ID: {
            "arrival": arrival_time,
            "serving": serving_time
            }
        }
        str_out = [json.dumps(
            pred, cls=NumpyEncoder) for pred in output]
        prediction_encoded = StringCodec.encode_output(
            payload=str_out, name="output")
        logger.info(f"Output:\n{prediction_encoded}\nwas sent!")
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs = [prediction_encoded],
            parameters=types.Parameters(times=str(times))
        )
