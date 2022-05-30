# TODO check all these



import os
import logging
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from copy import deepcopy

logger = logging.getLogger(__name__)

class VideoYolo(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False

    def load(self):
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.resnet = models.resnet101(pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        # TODO some transformation might be needed
        objs = self.model(X)
        output = self.get_cropped(objs)
        logger.info(f"Output:\n{output}\nwas sent!")
        return output

    def get_cropped(self, objs):
        # TODO 1. get cropped
        # TODO selected calsses subject to change based on the extra
        # information on the pipeline
        liscense_labels = ['car', 'truck']
        car_labels = ['car']
        person_labels = ['person']
        output_list = {'people': [], 'car': [], 'liscense': []}
        objs = objs.crop()
        for obj in objs:
            for label in liscense_labels:
                if label in obj['label']:
                    output_list['liscense'].append(deepcopy(obj))
                    break
            for label in car_labels:
                if label in obj['label']:
                    output_list['car'].append(deepcopy(obj))
                    break
            for label in person_labels:
                if label in obj['label']:
                    output_list['person'].append(deepcopy(obj))
                    break
        # 2. check if there are person or car in the cropped
        # 3. return the results as people and person keys to the next step (list for each key)
        # 4. make it like the format
        return output_list
