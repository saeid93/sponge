import logging
import os
import numpy as np
import torch 

class CascadeInception(object):
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        
    def predict(self, X, features_names=None):
        return X*3
