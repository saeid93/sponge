import logging
import os
import numpy as np

class NodeOne:
    def predict(self, X: np.array, features):
        return X*2