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
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput)
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
import json

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")


# PREDICTIVE_UNIT_ID = 'name' #os.environ['PREDICTIVE_UNIT_ID']

class ResnetHuman(MLModel):
    async def load(self) -> bool:
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'resnet101'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
        logger.error('Init function complete!')
        model = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        logger.error('Loading the ML models')
        # TODO cpu and gpu from env variable
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.error('request input:\n')
            # logger.error(f"{request_input}\n")
            decoded_inputs = self.decode(request_input)
            # logger.error('decoded_input:\n')
            # logger.error(f"{list(decoded_inputs)}\n")
            X = []
            former_steps_timings = []
            for decoded_input in decoded_inputs:
                json_inputs = json.loads(decoded_input)
                former_steps_timings.append(json_inputs['time'])
                X.append(json_inputs['output']['person'])
        if X == []:
            return []
        logger.error(f"len(X):\n{len(X)}\n")
        logger.error(f"len(X[0])):\n{len(X[0])}\n")
        X_trans = [
            self.transform(
                Image.fromarray(
                    np.array(image).astype(np.uint8))) for image in X[0]] # TEMP X[0]
        batch = torch.stack(X_trans, axis=0)
        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1) * 100
        percentages = percentages.detach().numpy()
        image_net_class = np.argmax(percentages, axis=1)
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}": arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}": serving_time,
        }
        # timing.update(former_steps_timing)
        output = {
            "time": timing,
            "output": image_net_class.tolist()
        }
        logger.error(f"Output:\n{output}\nwas sent!")
        return output
