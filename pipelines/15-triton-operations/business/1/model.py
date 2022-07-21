from requests import request
import triton_python_backend_utils as pb_utils
import asyncio
import logging

class TritonPythonModel:

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    def execute(self, requests):
      
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make sure
      # to replace `inputs` argument with a list of `pb_utils.Tensor` objects.
      infer_responses = []
      counter = 0
      logging.info("len of request: ", len(requests))
      for request in requests:
          counter += 1
          logging.info('hi')
          inference_request = pb_utils.InferenceRequest(
          model_name='resnet50',
          requested_output_names=['output'],
          inputs=[pb_utils.get_input_tensor_by_name(request, "input")])

          inference_response = inference_request.exec()
          infer_responses.append(inference_response)
          logging.info("salammmmmm")
          if counter > 5:
              break

      
      responses = []

      for infer_response in infer_responses:
        # Check if the inference response has an error
        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output1 = pb_utils.get_output_tensor_by_name(infer_response, 'output')
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output1])
            responses.append(inference_response)
            # Decide the next steps for model execution based on the received output
            # tensors.
        print("done")
        return [responses, "salam"]