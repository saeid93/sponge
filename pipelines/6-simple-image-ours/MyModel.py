import io
import logging
import numpy as np
from PIL import Image
logger = logging.getLogger('__mymodel__')
class MyModel(object):
  def __init__(self):
    logger.info("initializing...")
    logger.info("load model here...")
    self._model = None
    logger.info("model has been loaded and initialized...")
  def predict(self, X, features_names):
    """ Seldon Core Prediction API """
 
    logger.info("predict called...")
    # Use Pillow to convert to an RGB image then reverse channels.
    logger.info('converting tensor to image')
    img = Image.open(io.BytesIO(X)).convert('RGB')
    img = np.array(img)
    img = img[:,:,::-1]
    logger.info("image size = {}".format(img.shape))
    if self._model:
      logger.info("perform inference here...")
    # This will serialize the image into a JSON tensor
    logger.info("returning prediction...")
    # Return the original image sent in RGB
    return img[:,:,::-1]