import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import time
from mlserver import MLModel
import numpy as np
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver import MLModel
from typing import List

# TODO balooning has not been implemented yet

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.info(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'resnet-human'
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

def decode_from_bin(
    inputs: List[bytes], shapes: List[
        List[int]], dtypes: List[str]) -> List[np.array]:
    batch = []
    for input, shape, dtype in zip(inputs, shapes, dtypes):
        buff = memoryview(input)
        array = np.frombuffer(buff, dtype=dtype).reshape(shape)
        # array = [array] # TEMP workaround
        batch.append(array)
    return batch

class ResnetHuman(MLModel):
    async def load(self) -> bool:
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'resnet18'
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        logger.info(f'max_batch_size: {self._settings.max_batch_size}')
        logger.info(f'max_batch_time: {self._settings.max_batch_time}')
        self.batch_size = self._settings.max_batch_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
        logger.info('Init function complete!')
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
        logger.info('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            prev_nodes_times = request_input.parameters.times
            if type(prev_nodes_times) == str:
                logger.info(f"prev_nodes_times:\n{prev_nodes_times}")
                prev_nodes_times = [eval(eval(prev_nodes_times)[0])]
            else:
                prev_nodes_times = list(
                    map(lambda l: eval(eval(l)[0]), prev_nodes_times))
            # dtypes = request_input.parameters.dtype
            shapes = request_input.parameters.datashape
            batch_shape = request_input.shape[0]
            dtypes = batch_shape * ['u1']
            # batch one edge case
            logger.info(shapes)
            logger.info(type(shapes))
            logger.info(dtypes)
            logger.info(type(dtypes))
            if type(shapes) != list:
                shapes = eval(shapes)
                # dtypes = eval(dtypes)
            else:
                logger.info(f"shapes:\n{shapes}")
                shapes = [[253, 294, 3]] * 5 # TEMP hack
                # shapes = list(map(lambda l: eval(l), shapes))
            input_data = request_input.data.__root__
            X = decode_from_bin(
                inputs=input_data, shapes=shapes, dtypes=dtypes)
        self.request_counter += batch_shape
        self.batch_counter += 1

        # preprocessings
        converted_images = [Image.fromarray(
            np.array(image, dtype=np.uint8)) for image in X]
        X_trans = [self.transform(
            converted_image) for converted_image in converted_images]
        batch = torch.stack(X_trans, axis=0)

        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1) * 100
        percentages = percentages.detach().numpy()
        image_net_class = np.argmax(percentages, axis=1)
        output = image_net_class.tolist()
        logger.info(f"{image_net_class=}")
        serving_time = time.time()
        times = {
            PREDICTIVE_UNIT_ID: {
            "arrival": arrival_time,
            "serving": serving_time
            }
        }
        this_node_times = [times] * batch_shape
        times = []
        for this_node_time, prev_nodes_time in zip(
            this_node_times, prev_nodes_times):
            this_node_time.update(prev_nodes_time)
            times.append(this_node_time)
        batch_times = list(map(lambda l: str(l), times))
        if self.batch_size == 1:
            batch_times = str(batch_times)

        # processing inference response
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="int",
                    shape=[batch_shape],
                    datatype="INT32",
                    data=output,
                    parameters=Parameters(
                        times=batch_times,
                        content_type='np'
                    ),
            )],
            model_name=self.name,
            parameters=Parameters(
                type_of='int'
            )
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload

