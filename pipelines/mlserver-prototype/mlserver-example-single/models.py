from itertools import count
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
import json
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
import time
from mlserver.codecs import (
    StringCodec,
)
# import logging

# logger = logging.getLogger("code.simple")

_to_exclude = {
    "parameters": {DecodedParameterName, "headers"},
    'inputs': {"__all__": {"parameters": {DecodedParameterName, "headers"}}}
}

def model(input):
  time.sleep(1)
  output =  input
  return output

class NodeOne(MLModel):
  async def load(self) -> bool:
    self._model = model
    self.ready = True
    self.counter = 0
    logger.error("This is from the logger")
    return self.ready

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
      outputs = []
      self.counter += 1
      logger.error(f"counter: {self.counter}")
      logger.error('*'*50)
      logger.error("This is recieved request")
      logger.error(payload.inputs)
      for request_input in payload.inputs:
          decoded_input = self.decode(request_input)
          logger.error(f"decoded_input:\n{decoded_input}")
          logger.error(f"size of the input: {np.shape(decoded_input)}")
          model_output = self._model(decoded_input)
          logger.error("*"*50)
          logger.error("model_output item:\n")
          logger.error(model_output[0])
          logger.error("model_output:\n")
          # logger.error(model_output.tolist())
          logger.error("model_output type:\n")
          logger.error(type(model_output))
          outputs.append(
              ResponseOutput(
                  name=request_input.name,
                  datatype=request_input.datatype,
                  # parameters={
                  #   "content_type": "np"
                  # },
                  shape=request_input.shape,
                  data=model_output
              )
          )
      logger.error(outputs)
      return InferenceResponse(model_name=self.name, outputs=outputs)
