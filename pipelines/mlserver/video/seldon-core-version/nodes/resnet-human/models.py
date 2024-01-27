import os
import json
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
    Parameters,
)
from mlserver import MLModel
from typing import List
from fastapi import Request, Response
from mlserver.handlers import custom_handler

# TODO balooning has not been implemented yet

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "resnet-human"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


def decode_from_bin(
    inputs: List[bytes], shapes: List[List[int]], dtypes: List[str]
) -> List[np.array]:
    batch = []
    for input, shape, dtype in zip(inputs, shapes, dtypes):
        buff = memoryview(input)
        array = np.frombuffer(buff, dtype=dtype).reshape(shape)
        batch.append(array)
    return batch


try:
    USE_THREADING = os.getenv("USE_THREADING", "False").lower() in ("true", "1", "t")
    logger.info(f"USE_THREADING set to: {USE_THREADING}")
except KeyError as e:
    USE_THREADING = False
    logger.info(
        f"USE_THREADING env variable not set, using default value: {USE_THREADING}"
    )

try:
    NUM_INTEROP_THREADS = int(os.environ["NUM_INTEROP_THREADS"])
    logger.info(f"NUM_INTEROP_THREADS set to: {NUM_INTEROP_THREADS}")
except KeyError as e:
    NUM_INTEROP_THREADS = 1
    logger.info(
        f"NUM_INTEROP_THREADS env variable not set, using default value: {NUM_INTEROP_THREADS}"
    )

try:
    NUM_THREADS = int(os.environ["NUM_THREADS"])
    logger.info(f"NUM_THREADS set to: {NUM_THREADS}")
except KeyError as e:
    NUM_THREADS = 1
    logger.info(f"NUM_THREADS env variable not set, using default value: {NUM_THREADS}")

if USE_THREADING:
    torch.set_num_interop_threads(NUM_INTEROP_THREADS)
    torch.set_num_threads(NUM_THREADS)


try:
    LOGS_ENABLED = os.getenv("LOGS_ENABLED", "True").lower() in ("true", "1", "t")
    logger.info(f"LOGS_ENABLED set to: {LOGS_ENABLED}")
except KeyError as e:
    LOGS_ENABLED = True
    logger.info(
        f"LOGS_ENABLED env variable not set, using default value: {LOGS_ENABLED}"
    )

if not LOGS_ENABLED:
    logger.disabled = True


class ResnetHuman(MLModel):

    @custom_handler(rest_path="/change")
    async def change_thread(self, request: Request) -> Response:
        """
        changing the intra and inter thread
        """
        raw_data = await request.body()
        threads_parameters = json.loads(raw_data.decode("utf-8"))
        # USE_THREADING = True
        if USE_THREADING:
            # torch.set_num_interop_threads(threads_parameters["interop_threads"])
            torch.set_num_threads(threads_parameters["num_threads"])
        logger.info(f"changed interop_threads to {torch.get_num_interop_threads()}")
        logger.info(f"changed num_threads to {torch.get_num_threads()}")
        # await self.load()

    async def load(self) -> bool:
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ["MODEL_VARIANT"]
            logger.info(f"MODEL_VARIANT set to: {self.MODEL_VARIANT}")
        except KeyError as e:
            self.MODEL_VARIANT = "resnet18"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}"
            )
        logger.info(f"max_batch_size: {self._settings.max_batch_size}")
        logger.info(f"max_batch_time: {self._settings.max_batch_time}")
        self.batch_size = self._settings.max_batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logger.info("Init function complete!")
        model = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        logger.error("Loading the ML models")
        # TODO cpu and gpu from env variable
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.default_shape = [253, 294, 3]
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info("model loading complete!")
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        if not LOGS_ENABLED:
            logger.disabled = True
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            batch_shape = request_input.shape[0]
            if batch_shape == 1:
                dtypes = [request_input.parameters.extended_parameters['dtype']]
                shapes = [request_input.parameters.extended_parameters['datashape']]
                prev_node_name = request_input.parameters.extended_parameters['node_name']
                prev_arrival = request_input.parameters.extended_parameters['arrival']
                prev_serving = request_input.parameters.extended_parameters['serving']
            else:
                extended_parameters_repeated = request_input.parameters.extended_parameters
                if request_input.parameters.extended_parameters is None:
                    extended_parameters_repeated = request_input.parameters.extended_parameters_repeated
                else:
                    extended_parameters_repeated = request_input.parameters.extended_parameters
                dtypes = list(map(lambda l: l['dtype'], extended_parameters_repeated))
                shapes = list(map(lambda l: l['datashape'], extended_parameters_repeated))
                # NOTE the assumption here is that the requests in a bactch always come from the
                # same path
                prev_node_name = extended_parameters_repeated[0]['node_name']
                prev_arrival = extended_parameters_repeated[0]['arrival']
                prev_serving = extended_parameters_repeated[0]['serving']
            input_data = request_input.data.__root__
            logger.info(f"shapes:\n{shapes}")
            X = decode_from_bin(inputs=input_data, shapes=shapes, dtypes=dtypes)
        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += batch_shape
        self.batch_counter += 1
        # preprocessings
        converted_images = [
            Image.fromarray(np.array(image, dtype=np.uint8)) for image in X
        ]
        X_trans = [
            self.transform(converted_image) for converted_image in converted_images
        ]
        batch = torch.stack(X_trans, axis=0)

        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1) * 100
        percentages = percentages.detach().numpy()
        image_net_class = np.argmax(percentages, axis=1)
        output = image_net_class.tolist()
        logger.info(f"{image_net_class=}")
        serving_time = time.time()
        next_node = "out"
        extended_parameters = {
            "node_name": prev_node_name + [PREDICTIVE_UNIT_ID],
            "arrival": prev_arrival + [arrival_time],
            "serving": prev_serving + [serving_time],
            "next_node": next_node}
        batch_extended_parameters = [extended_parameters] * batch_shape
        if batch_shape == 1:
            batch_extended_parameters = extended_parameters
            parameters = {"extended_parameters": batch_extended_parameters}
        elif self.settings.max_batch_size != 1:
            parameters = {"extended_parameters": batch_extended_parameters}            
        else:
            parameters = {"extended_parameters_repeated": batch_extended_parameters}
        # processing inference response
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="int",
                    shape=[batch_shape],
                    datatype="INT32",
                    data=output,
                    parameters=Parameters(
                        **parameters, content_type="np"
                    ),
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="int"),
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload
