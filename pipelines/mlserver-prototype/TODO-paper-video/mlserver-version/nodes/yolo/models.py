import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import time
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
from copy import deepcopy
# import torchvision
import sys
sys.path.insert(0, './cache/ultralytics_yolov5_master')

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

class Yolo(MLModel):
    async def load(self):
        self.loaded = False
        self.counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'yolov5s' 
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            logger.error('Loading the ML models')
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            torch.hub.set_dir('./cache')
            self.model = torch.hub.load('ultralytics/yolov5', self.MODEL_VARIANT)
            logger.error('model loaded!')
            self.loaded = True
            logger.error('model loading complete!')
        except OSError:
            pass


    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        outputs = []
        self.counter += 1
        logger.error(f"counter: {self.counter}")
        logger.error('*'*50)
        # logger.error("This is recieved request")
        # logger.error(payload.inputs)
        for request_input in payload.inputs:
            decoded_input = self.decode(request_input)
            # logger.error(f"decoded_input:\n{decoded_input}")
            logger.error(f"type of decoded input: {type(decoded_input)}")
            logger.error(f"size of the input: {np.shape(decoded_input)}")

        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        X = decoded_input
        logger.error(f"Incoming input:\n{X}\nwas recieved!")
        logger.error(f"input type: {type(X)}")
        logger.error(f"input shape: {X.shape}") 
        X = np.array(X, dtype=np.uint8)
        logger.error(f"output type: {type(X)}")
        logger.error(f"output shape: {X.shape}")
        objs = self.model(X)
        serving_time = time.time()
        output = self.get_cropped(objs)
        logger.error(f"arrival time {PREDICTIVE_UNIT_ID}: {arrival_time}")
        logger.error(f"serving time {PREDICTIVE_UNIT_ID}: {serving_time}")
        # output[f'arrival_{PREDICTIVE_UNIT_ID}'] = arrival_time
        # output[f'serving_{PREDICTIVE_UNIT_ID}'] = serving_time
        output['time'] = {
            f'arrival_{PREDICTIVE_UNIT_ID}': arrival_time,
            f'serving_{PREDICTIVE_UNIT_ID}': serving_time
        }
        logger.error(f"Output:\n{output}\nwas sent!")
        return output

    def get_cropped(self, result):
        """
        crops selected objects for
        the subsequent nodes
        """
        result = result.crop()
        liscense_labels = ['car', 'truck']
        car_labels = ['car']
        person_labels = ['person']
        output_list = {'person': [], 'car': [], 'liscense': []}
        for obj in result:
            for label in liscense_labels:
                if label in obj['label']:
                    output_list['liscense'].append(deepcopy(obj['im'].tolist()))
                    break
            for label in car_labels:
                if label in obj['label']:
                    output_list['car'].append(deepcopy(obj['im'].tolist()))
                    break
            for label in person_labels:
                if label in obj['label']:
                    output_list['person'].append(deepcopy(obj['im'].tolist()))
                    break
        return output_list
