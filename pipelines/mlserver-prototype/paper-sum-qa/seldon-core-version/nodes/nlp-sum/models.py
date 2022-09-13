import os
import time
import json
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput, Parameters
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
from copy import deepcopy
from transformers import pipeline

import sys
sys.path.insert(0, './cache/ultralytics_yolov5_master')

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

class GeneralNLP(MLModel):
    async def load(self):
        self.loaded = False
        self.counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'sshleifer/distilbart-cnn-12-6'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.error(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'summarization' 
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        logger.error('Loading the ML models')
        self.model  = pipeline(task=self.TASK, model=self.MODEL_VARIANT)
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        former_steps_timing = None
        arrival_time = time.time()
        for request_input in payload.inputs:
            decoded_input = self.decode(request_input)
            X = decoded_input[0]
            logger.error(f"type of decoded input:\n{type(X)}")
            logger.error(f"size of the input:\n{np.shape(X)}")
            logger.error(f"input:\n{X}")
            if type(X) is not str: # If not the first node TODO use another check
                former_steps_timing = X['time']
                if type(X) is dict and 'label' in X.keys():    
                    label = X['label'] # To be later used for node logic
                    X = X['input']
                else:
                    logger.info("Here!!")
                    X = X['output']
                    X = list(X.values())[0]
        logger.error(f"sending\n{X}\nto the model")
        output = self.model(X)
        logger.info(f"model output:\n{output}")
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        if former_steps_timing is not None:
            timing.update(former_steps_timing)
        if self.TASK == "text-classification":
            output = {
                'time': timing,
                'label': output[0]['label'],
                'input': X}
        else:
            output = {
                'time': timing,
                'output': output[0],                
            }
        logger.error(f"Output:\n{output}\nwas sent!")
        response_bytes = json.dumps(output).encode("UTF-8")        
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
