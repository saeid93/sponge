from itertools import count
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
import json
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
import logging

logger = logging.getLogger("code.simple")

# breakpoint()
_to_exclude = {
    "parameters": {DecodedParameterName, "headers"},
    'inputs': {"__all__": {"parameters": {DecodedParameterName, "headers"}}}
}
class NodeOne(MLModel):
  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    # self.model_uri = await get_model_uri(self._settings)
    # settings, models_settings = await load_settings()
    self._model = lambda l: 2*l
    self.ready = True
    self.counter = 0
    logger.debug("This is from the logger")
    print('model_loaded')
    # settings, models_settings = await load_settings('/mnt/models')
    # print(f"settings:\n{settings}")
    # print(f"_settings:\n{self._settings}")
    # print(f"models_settings:\n{models_settings}")
    return self.ready

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
      outputs = []
      self.counter += 1
      print(payload.inputs)
      # print(f'type: {type(payload.inputs)}')
      # print(f'lenght: {len(payload.inputs)}')
      print(f"counter: {self.counter}")
      # print('\n')
      for request_input in payload.inputs:
        #   print('-' * 50)
        #   print(np.shape(payload))
          decoded_input = self.decode(request_input)
        #   as_dict = request_input.dict(exclude=_to_exclude)
        #   print(json.dumps(as_dict, indent=2))
          model_output = self._model(decoded_input)
        #   print(type(request_input.data))
          outputs.append(
              ResponseOutput(
                  name=request_input.name,
                  datatype=request_input.datatype,
                  shape=request_input.shape,
                  data=model_output.tolist()
              )
          )
      
      return InferenceResponse(model_name=self.name, outputs=outputs)
