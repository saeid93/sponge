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
        self.counter = 0
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
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

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
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        arrival_time = time.time()
        # former_steps_timing = X['time']
        if X['person'] == []:
            return []
        X = X['person']
        # if self.WITH_PREPROCESSOR:
        #     X_trans = torch.from_numpy(X.astype(np.float32))
        # else:
        #     # X_trans = Image.fromarray(X.astype(np.uint8))
        #     # X_trans = self.transform(X_trans)
        X_trans = [
            self.transform(
                Image.fromarray(
                    np.array(image).astype(np.uint8))) for image in X]
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
