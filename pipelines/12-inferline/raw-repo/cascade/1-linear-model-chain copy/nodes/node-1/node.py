import torchvision
import torch
import pandas as pd
import numpy as np
from torchvision import transforms


class NodeOne(object):
    def __init__(self) -> None:
        # standard resnet image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),                    
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]                  
            )])
    
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet.eval()

    def predict(self, X, features_names=None):
        return X*2
