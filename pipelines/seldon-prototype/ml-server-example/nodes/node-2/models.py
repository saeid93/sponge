from mlserver import MLModel
import numpy as np
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, Parameters

class MyCustomRuntime(MLModel):

  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    # self.model_uri = await get_model_uri(self._settings)
    self._model = lambda l: 3*l
    self.ready = True
    return self.ready

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    # TODO: Replace for custom logic to run inference
    outputs = self._model(payload)
    return InferenceResponse(
      model_name='fake',
      # Include any actual outputs from inference
      outputs=[],
      parameters=Parameters(headers={"model_uri": "fake"})
    )