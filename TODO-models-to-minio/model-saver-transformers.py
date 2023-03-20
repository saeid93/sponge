import os
import sys
from transformers import pipeline

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..')))

from experiments.utils.constants import (
    ACCURACIES_PATH,
    TEMP_MODELS_PATH
)
import shutil

# TODO add loading from some yaml

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

# TODO substitute with minio copying
# TODO check if the directory exist

def upload_minio(bucket_name: str):
    """uploads model files to minio
        and removes them from the disk

    Args:
        bucket_name (str): name of the minio bucket
    """
    output_dir = os.path.join(TEMP_MODELS_PATH, bucket_name)
    # copy generated models to minio
    os.system(f"mc mb minio/{bucket_name} -p")
    os.system(f"mc cp -r {output_dir}"
              f" minio/{bucket_name}")
    shutil.rmtree(output_dir)

os.system(f"sudo mv ./{dirname} /mnt/myshareddir/huggingface/")
