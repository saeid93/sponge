# %%
import os

# %%
VERSION='22.05'
os.system(f"docker pull nvcr.io/nvidia/tritonserver:{VERSION}-py3")
# add --gpus=<number of gpus> on gpu machines
# add -d to run at background and going to the next cell
os.system("docker run --rm -d -p5000:8000 -p5001:8001 -p5002:8002"
          f" -v {os.getcwd()}/models:/models "
          f"nvcr.io/nvidia/tritonserver:{VERSION}-py3"
          " tritonserver --model-repository=/models")

# %%
text = "Replace me by any text you'd like."
from transformers import DistilBertTokenizer, DistilBertModel
tokenizer_dis = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_dis

# %%
encoded_input = tokenizer_dis(text, return_tensors='pt')
encoded_input.numpy()

# %%
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:5000', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))

model_name = "distiluncase"
inputs = []
inputs.append(
    httpclient.InferInput(
        name="input", shape=encoded_input.shape, datatype="INT64")
)
inputs[0].set_data_from_numpy(encoded_input, binary_data=False)
 
outputs = []
outputs.append(httpclient.InferRequestedOutput(name="output"))
 
result = triton_client.infer(
    model_name=model_name, inputs=encoded_input, outputs=outputs)
triton_client.close()

# %%
# use onnx model
import onnx
import onnxruntime
model_dir = "models/distiluncase/1" 
model_path = os.path.join(model_dir, "model.onnx")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
 
ort_session = onnxruntime.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=['CPUExecutionProvider'])


# %%
onnx_model

# %%



