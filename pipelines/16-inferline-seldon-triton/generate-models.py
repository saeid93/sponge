from typing import List
import os
import timm
import torch
import shutil
# add clean path

PATH=(
    "/home/cc/infernece-pipeline-joint-optimization"
    "/pipelines/16-inferline-seldon-triton/")

BUCKET_NAME="seldon"


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
    versions: List[list]):
    # model name
    if source == 'timm':
        models_list = timm.list_models(model_name+'*', pretrained=True)
    else:
        models_list = torch.hub.list(source)
    model_path = os.path.join(
        PATH,
        BUCKET_NAME,
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
    versions: List[List[str]]):
    assert len(model_names) == len(versions),\
        "length modes list {} does not match versions list {}".fromat(
            len(model_names),
            len(versions)
        )
    for source, model_name, version in zip(source, model_names, versions):
        generate_model_variants(
            source=source,
            model_name=model_name,
            versions=version
        )

# read these from json/yamls build with a proper config builder
model_generator(
    source = ['timm', 'ultralytics/yolov5'],
    model_names = ['resnet', 'yolo'],
    versions = [['18', '34', '101'], ['yolov5s', 'yolov5n', 'yolov5x6']]
)

# copy generated models to minio
os.system(f"mc mb minio/{BUCKET_NAME} -p")
os.system(f"mc cp -r {PATH}{BUCKET_NAME} minio/{BUCKET_NAME}")
shutil.rmtree(os.path.join(PATH, BUCKET_NAME))
