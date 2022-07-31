import os
from jinja2 import Environment, FileSystemLoader

svc_vars = [
    {"name": "Sandrine",  "score": 100},
    {"name": "Gergeley", "score": 87},
    {"name": "Frieda", "score": 92},
]

environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template("message.txt")

for student in students:
    filename = f"message_{student['name'].lower()}.txt"
    content = template.render(
        student,
        max_score=max_score,
        test_name=test_name
    )
    with open(filename, mode="w", encoding="utf-8") as message:
        message.write(content)
        print(f"... wrote {filename}")

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



