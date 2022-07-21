import logging
import os

import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class Resnet(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        try:
            self.WITH_PREPROCESSOR = bool(os.environ['WITH_PREPROCESSOR'])
            logging.info(f'WITH_PREPROCESSOR set to: {self.WITH_PREPROCESSOR}')
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
            logging.info(f'Preprocessor loaded!')
        except KeyError as e:
            self.WITH_PREPROCESSOR = False
            logging.warning(
                f"WITH_PREPROCESSOR env variable not set, using default value: {self.WITH_PREPROCESSOR}")
        logger.info('Init function complete!')

    def load(self):
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input with shape:\n{X.shape}\nwas recieved!")
        logger.info(f"recived intput type: {type(X)}")
        # logger.info(f"recieved intput element type: {type(X[0].keys())}")
        yolo_labels = []
        yolo_images = []
        trans_yolo_images = []
        for yolo_output in X:
            yolo_labels.append(yolo_output['label'])
            yolo_images.append(np.array(yolo_output['im']))
        # logger.info(f"image shape: {yolo_images[0].shape}")
        if yolo_labels == []:
            output = {
                'yolo_labels': [],
                'resnet_classes': []}
        else:
            if self.WITH_PREPROCESSOR:
                batch = torch.stack(
                    list(map(
                        lambda a: self.transform(
                            Image.fromarray(a.astype(np.uint8))), yolo_images)))
                logger.info(f"batch shape: {batch.shape}")    
            else:
                trans_yolo_images = list(
                    map(lambda a: torch.tensor(a), yolo_images)
                )
                batch = torch.stack(trans_yolo_images)
            out = self.resnet(batch)
            percentages = torch.nn.functional.softmax(out, dim=1) * 100
            percentages = percentages.detach().numpy()
            logger.info(f'Percentages shape: {percentages.shape}')
            # max_prob_percentage = max(percentages)
            max_prob_class = percentages.argmax(axis=1).tolist()
            output = {
                'yolo_labels': yolo_labels,
                'resnet_classes': max_prob_class}
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
