import argparse
import sys
import numpy as np
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()
    model_name = "intel_image_class"

    image_path = '1.jpeg'
    image = np.asarray(Image.open(image_path).resize((100, 100)))
    image = np.expand_dims(image, axis=0)
    image = np.divide(image, 255.0).astype("float32")
    print(image.shape)
    inputs = []
    inputs.append(
        httpclient.InferInput(name="input_1", shape=image.shape, datatype="FP32")
    )
    inputs[0].set_data_from_numpy(image, binary_data=False)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput(name="dense"))

    result = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    triton_client.close()
    print(result.as_numpy("dense"))