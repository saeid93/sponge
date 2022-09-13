import os
import time

from itertools import count
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
import json
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
from transformers import pipeline

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

class GeneralNLP(MLModel):

    async def load(self):
        logger.error('Loading the ML models')
        self.model  = pipeline(task=self.TASK, model=self.MODEL_VARIANT)
        self.loaded = True
        logger.error('model loading complete!')
        self.loaded = False
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = "distilbert-base-uncased"
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.error(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'question-answering'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        try:
            self.CONTEXT = os.environ['CONTEXT']
            logger.error(f'CONTEXT set to: {self.CONTEXT}')
        except KeyError as e:
            self.CONTEXT = 'default context'
            logger.error(
                f"CONTEXT env variable not set, using default value: {self.CONTEXT}")

    async def predict(self, payload, features_names=None):
        if self.loaded == False:
            self.load()
        logger.error(f'Incoming input type: {type(X)}')
        logger.error(f"Incoming input:\n{X}\nwas recieved!")
        arrival_time = time.time()
        for request_input in payload.inputs:
            decoded_input = self.decode(request_input)
            logger.error(f"type of decoded input: {type(decoded_input)}")
            logger.error(f"size of the input: {np.shape(decoded_input)}")
            X = decoded_input
            # TODO batching considerations
            former_steps_timing = X['time']
            X = X['output']['summary_text']
            qa_input = {
                'question': X,
                'context': self.CONTEXT
            }
        output = self.model(qa_input)
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        timing.update(former_steps_timing)
        output = {
            'time': timing,
            'output': output
        }
        logger.error(f"Output:\n{output}\nwas sent!")
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(
                    name="echo_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=Parameters(content_type="str")
                )
            ]
        )
