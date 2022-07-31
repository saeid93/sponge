import timm
import torch
import click
import yaml
import shutil
models_list = timm.list_models('vit'+'*', pretrained=True)
print(models_list)