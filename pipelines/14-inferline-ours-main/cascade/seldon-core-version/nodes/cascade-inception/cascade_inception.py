import logging
import os

import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class CascadeInception(object):
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
        logger.info('Init function complete!')

    def load(self):
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.inception_v3(pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, last_step_output, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"type X : {type(last_step_output['X'])}")
        logger.info(f"shape of np X: {np.array(last_step_output['X']).shape}")
        logger.info(f"input keys: {last_step_output.keys()}")
        for k, v in last_step_output.items():
            logging.info(f"type {k}: {type(v)}")
        # logger.info(f"input values: {X.values()}")
        # logger.info(f"Incoming input:\n{X}\nwas recieved!")
        # X = Image.fromarray(X.astype(np.uint8))
        # X = self.transform(X)
        # batch = torch.unsqueeze(X, 0)
        # out = self.resnet(batch)
        # _, indices = torch.sort(out, descending=True)
        # indices = indices.detach().numpy()[0]
        # percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # percentages = percentages.detach().numpy()
        # max_prob_percentage = max(percentages)
        # output = {
        #     'indices': list(map(int, list(indices))),
        #     'percentages': list(map(float, list(percentages))),
        #     'max_prob_percentage': float(max_prob_percentage),
        #     'model_name': 'inception'}
        # return output
        return []
