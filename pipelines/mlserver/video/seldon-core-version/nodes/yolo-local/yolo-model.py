import os
import torch
import time
import json
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
from copy import deepcopy
from typing import List
from fastapi import Request, Response
from mlserver.handlers import custom_handler


try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "yolo"
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

try:
    QUEUE_MODE = os.getenv("QUEUE_MODE", "True").lower() in ("true", "1", "t")
    logger.info(f"QUEUE_MODE set to: {QUEUE_MODE}")
except KeyError as e:
    QUEUE_MODE = False
    logger.info(f"QUEUE_MODE env variable not set, using default value: {QUEUE_MODE}")

if not LOGS_ENABLED:
    logger.disabled = True


class Yolo(MLModel):
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

    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ["MODEL_VARIANT"]
            logger.info(f"MODEL_VARIANT set to: {self.MODEL_VARIANT}")
        except KeyError as e:
            self.MODEL_VARIANT = "yolov5s"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}"
            )
        try:
            logger.info("Loading the ML models")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"max_batch_size: {self._settings.max_batch_size}")
            logger.info(f"max_batch_time: {self._settings.max_batch_time}")
            self.model = torch.hub.load("ultralytics/yolov5", self.MODEL_VARIANT)
            logger.info("model loaded!")
            self.loaded = True
            logger.info("model loading complete!")
        except OSError:
            raise ValueError("model loading unsuccessful")
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        if self.loaded == False:
            self.load()
        arrival_time = time.time()

        for request_input in payload.inputs:
            batch_shape = request_input.shape[0]
            if batch_shape == 1:
                dtypes = [request_input.parameters.extended_parameters["dtype"]]
                shapes = [request_input.parameters.extended_parameters["datashape"]]
            else:
                extended_parameters_repeated = (
                    request_input.parameters.extended_parameters
                )
                if request_input.parameters.extended_parameters is None:
                    extended_parameters_repeated = (
                        request_input.parameters.extended_parameters_repeated
                    )
                else:
                    extended_parameters_repeated = (
                        request_input.parameters.extended_parameters
                    )
                dtypes = list(map(lambda l: l["dtype"], extended_parameters_repeated))
                shapes = list(
                    map(lambda l: l["datashape"], extended_parameters_repeated)
                )
            input_data = request_input.data.__root__
            logger.info(f"shapes:\n{shapes}")
            X = decode_from_bin(inputs=input_data, shapes=shapes, dtypes=dtypes)
        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")
        objs = self.model(X)
        outputs = self.get_cropped(objs)
        serving_time = time.time()
        # TEMP Currently considering only one person per pic, zero index
        categories = ["person", "car", "liscense"]
        category_to_node = {
            "person": "resnet-human",
            "car": "resnet-car",
            "liscense": "resnet-liscense",
        }
        pics = []
        next_nodes = []
        for output in outputs:
            for category in categories:
                if output[category] != []:
                    pics.append(output[category][0])
                    next_nodes.append(category_to_node[category])
        output_data = list(map(lambda l: l.tobytes(), pics))
        dtypes = "u1"
        extended_parameters = {
            "node_name": [PREDICTIVE_UNIT_ID],
            "arrival": [arrival_time],
            "serving": [serving_time],
            "dtype": dtypes,
        }
        batch_extended_parameters = [extended_parameters] * batch_shape
        for pic, extended_parameters in zip(pics, batch_extended_parameters):
            # list of list to hande for the shape to be able to handle images with multipe outputs
            extended_parameters["datashape"] = list(pic.shape)
        for next_node, extended_parameters in zip(
            next_nodes, batch_extended_parameters
        ):
            if QUEUE_MODE:
                next_node = f"queue-{next_node}"
            logger.info(f"next_node: {next_node}")
            extended_parameters["next_node"] = next_node
        if batch_shape == 1:
            batch_extended_parameters = extended_parameters
            parameters = {"extended_parameters": batch_extended_parameters}
        elif self.settings.max_batch_size != 1:
            parameters = {"extended_parameters": batch_extended_parameters}
        else:
            parameters = {"extended_parameters_repeated": batch_extended_parameters}
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="image",
                    shape=[batch_shape],
                    datatype="BYTES",
                    data=output_data,
                    parameters=Parameters(**parameters),
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="image"),
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload

    def get_cropped(self, result):
        """
        crops selected objects for
        the subsequent nodes
        """
        output_list = []
        for res in result.tolist():
            res = res.crop(save=False)
            liscense_labels = ["car", "truck"]
            car_labels = ["car"]
            person_labels = ["person"]
            res_output = {"person": [], "car": [], "liscense": []}
            for obj in res:
                for label in liscense_labels:
                    if label in obj["label"]:
                        res_output["liscense"].append(deepcopy(obj["im"]))
                        break
                for label in car_labels:
                    if label in obj["label"]:
                        res_output["car"].append(deepcopy(obj["im"]))
                        break
                for label in person_labels:
                    if label in obj["label"]:
                        res_output["person"].append(deepcopy(obj["im"]))
                        break
            output_list.append(res_output)
        return output_list
