# %%
import os

#
# %%
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:30803', verbose=True
    )
except Exception as e:
    print("context creation failed: " + str(e))

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
inp = tokenizer("This is a sample", return_tensors="pt")
print(inp['input_ids'].numpy())
inputs = []
inputs.append(
    httpclient.InferInput(
        name="input_ids",shape=inp['input_ids'].shape, datatype="INT64"
)
)
inputs[0].set_data_from_numpy(inp['input_ids'].numpy(), binary_data=False)
 
inputs.append(
    httpclient.InferInput(
        name="attention_mask", shape=inp['attention_mask'].shape, datatype="INT64")
)
inputs[1].set_data_from_numpy(inp['attention_mask'].numpy())
 
outputs = []
outputs.append(httpclient.InferRequestedOutput(name="logits"))
 
result = triton_client.infer(
    model_name=model_name, inputs=inputs, outputs=outputs)
triton_client.close()

# # %%
# # use onnx model




