import logging
import os

import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class CascadeResnet(object):
    def __init__(self) -> None:
        super().__init__()
        # standard resnet image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        self.loaded = False
        try:
            self.THRESHOLD = int(os.environ['CASCADE_RESNET_THRESHOLD'])
            logging.info(f'THRESHOLD set to: {self.THRESHOLD}')
        except KeyError as e:
            self.THRESHOLD = 85
            logging.warning(
                f"THRESHOLD env variable not set, using default value: {self.THRESHOLD}")
        logger.info('Init function complete!')

    def load(self):
        logger.info('Loading the ML models')
        # try:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        X_trans = Image.fromarray(X.astype(np.uint8))
        X_trans = self.transform(X_trans)
        batch = torch.unsqueeze(X_trans, 0)
        out = self.resnet(batch)
        _, indices = torch.sort(out, descending=True)
        indices = indices.detach().numpy()[0]
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        percentages = percentages.detach().numpy()
        max_prob_percentage = max(percentages)
        if max_prob_percentage > self.THRESHOLD:
            output = {
                'indices': list(map(int, list(indices))),
                'percentages': list(map(float, list(percentages))),
                'max_prob_percentage': float(max_prob_percentage),
                'route': -2}
        else:
            output = {
                'X': X.tolist(),
                'indices': list(map(int, list(indices))),
                'percentages': list(map(float, list(percentages))),
                'max_prob_percentage': float(max_prob_percentage),
                'route': 0}
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
