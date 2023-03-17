import os
from transformers import pipeline


task = "automatic-speech-recognition"
model_name = 'facebook/s2t-small-librispeech-asr'
batch_size = 5

model  = pipeline(
    task=task,
    model=model_name,
    batch_size=batch_size
)
dirname = model_name
if model_name.index("/"):
    dirname = model_name[model_name.index("/") + 1:]
model.save_pretrained(f"./{dirname}")
os.system(f"sudo mv ./{dirname} /mnt/myshareddir/huggingface/")
