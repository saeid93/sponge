import requests
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import os
from image import create_batch_image, send_request
os.system('sudo umount -l ~/my_mounting_point')
os.system('cc-cloudfuse mount ~/my_mounting_point')
inputs = []

print(f"start batch 2")
batch =create_batch_image(2)
inputs.append(
                httpclient.InferInput(
                    name="input", shape=batch.shape, datatype="FP32")
            )
inputs[0].set_data_from_numpy(batch.numpy(), binary_data=False)

outputs = []
outputs.append(httpclient.InferRequestedOutput(name="output"))
send_request('resnet', '1' , inputs, outputs, 2)