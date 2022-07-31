import os
from jinja2 import Environment, FileSystemLoader

PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/18-outside-poc"

svc_vars = {"username": "minioadmin",  "password": "minioadmin"},

environment = Environment(
    loader=FileSystemLoader(os.path.join(
        PATH, "templates/")))
svc_template = environment.get_template("service.yaml")

filename = os.path.join(PATH, "service.yaml")
content = template.render(svc_vars)

with open(filename, mode="w", encoding="utf-8") as message:
    message.write(content)

# # %%
# apiVersion: machinelearning.seldon.io/v1
# kind: SeldonDeployment
# metadata:
#   name: image-models-yolo
# spec:
#   name: default
#   predictors:
#   - graph:
#       implementation: TRITON_SERVER
#       logger:
#         mode: all
#       modelUri: s3://separate-servers/separate-servers/yolo
#       envSecretRefName: seldon-init-container-secret
#       name: image-models-resnet # This should have the same name as the model inside
#       type: MODEL
#     name: default
#     replicas: 1
#   protocol: kfserving

# # %%
# !kubectl apply -f secret.yaml -n default
# !kubectl apply -f seldon-triton-yolo.yaml -n default

# %%



