import logging
import os

import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class ImageModel(object):
    def __init__(self) -> None:
        super().__init__()
        # standard resnet image transformation
        # self.transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )])
        self.loaded = False
        try:
            self.MODEL_NAME = os.environ['MODEL_NAME']
            logging.info(f'MODEL_NAME set to: {self.MODEL_NAME}')
        except KeyError as e:
            self.MODEL_NAME = 'resnet50'
            logging.warning(
                f"THRESHOLD env variable not set, using default value: {self.MODEL_NAME}")
        logger.info('Init function complete!')

    def load(self):
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    # def predict(self, X, features_names=None):
    def predict(self, X_trans, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input:\n{X_trans}\nwas recieved!")
        X_trans = torch.from_numpy(X_trans.astype(np.float32))
        batch = torch.unsqueeze(X_trans, 0)
        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        percentages = percentages.detach().numpy()
        max_prob_percentage = max(percentages)
        max_prob_class = np.argmax(percentages)
        output = {
            'max_prob_class': int(max_prob_class),
            'max_prob_percentage': float(max_prob_percentage),
            'model_name': 'resnet',
            'route': 0}
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
