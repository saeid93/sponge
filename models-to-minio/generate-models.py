from distutils.command.upload import upload
from typing import List
import os
import sys
import timm
import torch
import click
import yaml
import shutil


# TODO fix later for other models

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..')))

from utils.constants import (
    TEMP_MODELS_PATH,
    KUBE_YAMLS_PATH
    )

def config_builder(
  name: str, platform: str, max_batch_size: int, source: str):
  config = (f"name: \"{name}\"\n"
            f"platform: \"{platform}\"\n"
            f"max_batch_size: {str(max_batch_size)}")
  if source == 'timm':
        common_config="""
input [
    {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
    }
]
output [
    {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    }
]
version_policy: { all { }}
        """
  else:
        common_config="""
input [
    {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 1, 3, 640, 640 ]
    }
]
output [
    {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    }
]
version_policy: { all { }}
        """
  return config + common_config


def generate_model_variants(
    source: str,
    model_name: list,
    versions: List[list],
    bucket_name: str):
    # model name
    if source == 'timm':
        models_list = timm.list_models(model_name+'*', pretrained=True)
    else:
        models_list = torch.hub.list(source)
    model_path = os.path.join(
        TEMP_MODELS_PATH,
        bucket_name,
        model_name,
    )
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    # TODO no config for yolo for now - if used solve it by:
    # https://github.com/ultralytics/yolov5/search?q=triton&type=issues
    # or it's own onnx tool:
    # https://github.com/ultralytics/yolov5/blob/master/export.py
    config_path = os.path.join(
        model_path,
        'config.pbtxt')
    os.makedirs(model_path)
    if source == 'timm':
        config = config_builder(
            name=model_name,
            platform='onnxruntime_onnx',
            max_batch_size=100,
            source=source)
        with open(config_path, 'w') as f:
            f.write(config)
    # TODO add language models the same way
    for variant_id, model_variant in enumerate(versions):
        if source == 'timm':
            model_full_name = model_name + model_variant
            if not model_full_name in models_list:
                raise ValueError(
                    f"Model {model_full_name} does not exist"
                )
            model = timm.create_model(model_full_name, pretrained=True)
        else:
            if not model_variant in models_list:
                raise ValueError(
                    f"Model {model_full_name} does not exist"
                )
            model = torch.hub.load(source, model_variant)
            pass
        model.eval()
        model_variant_dir = os.path.join(model_path, str(variant_id+1))
        model_variant_path = os.path.join(model_variant_dir, 'model.onnx')
        # if 'models' not in os.listdir("./"):
        os.makedirs(model_variant_dir)
        if source == 'timm':
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model, dummy_input,
                model_variant_path,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}})
        else:
            dummy_input = torch.randn(1, 3, 640, 640)
            torch.onnx.export(
               model,
               dummy_input,
               f=model_variant_path,
               input_names=['input'],
               output_names=['output'],
               dynamic_axes={
                   'images': {
                       0: 'batch',
                       2: 'height',
                       3: 'width'},  # shape(1,3,640,640)
                   'output': {
                       0: 'batch',
                       1: 'anchors'}  # shape(1,25200,85)
               })        

def model_generator(
    source: List[str],
    model_names: List[str],
    versions: List[List[str]],
    bucket_name: str):
    assert len(model_names) == len(versions),\
        "length modes list {} does not match versions list {}".fromat(
            len(model_names),
            len(versions)
        )
    for source, model_name, version in zip(
        source, model_names, versions):
        generate_model_variants(
            source=source,
            model_name=model_name,
            versions=version,
            bucket_name=bucket_name
        )

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


@click.command()
@click.option('--config-file', type=str, default='model-load')
def main(config_file: str):
    config_file_path = os.path.join(
        KUBE_YAMLS_PATH, f"{config_file}.yaml")
    with open(config_file_path, 'r') as cf:
        config = yaml.safe_load(cf)
    model_generator(**config)
    upload_minio(bucket_name='minio-seldon')

if __name__ == "__main__":
    main()
