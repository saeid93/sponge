import logging
import os
import numpy as np

class NodeTwo(object):
    def predict(self, X, features_names=None):
        return X*3
