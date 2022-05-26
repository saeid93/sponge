import logging
import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class EnsembleAlexnet(object):
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
        self.resnet = models.alexnet(pretrained=True)
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
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        percentages = percentages.detach().numpy()
        output = {
            'percentages': percentages.tolist(),
            'model_name': 'alexnet'
            }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
