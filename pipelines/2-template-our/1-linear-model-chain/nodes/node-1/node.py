import logging
import os
import numpy as np

class NodeOne(object):
    def predict(self, X, features_names=None):
        return X*2
