import os
from jinja2 import Environment, FileSystemLoader

PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/18-outside-poc/triton-inferline"


def setup_minio_secret(username: str, password: str):
    svc_vars = {"username": username,  "password": password}
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template("secret.yaml")
    filename = os.path.join(PATH, "secret.yaml")
    content = svc_template.render(svc_vars)
    with open(filename, mode="w", encoding="utf-8") as message:
        message.write(content)
    os.system(f"kubectl apply -f {PATH}/secret.yaml -n default")

def setup_triton_single(model_name: str, model_uri: str):
    svc_vars = {"model_name": model_name,  "model_uri": model_uri}
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template("triton-single.yaml")
    filename = os.path.join(PATH, f"triton-single-{model_name}.yaml")
    content = svc_template.render(svc_vars)
    with open(filename, mode="w", encoding="utf-8") as message:
        message.write(content)
    os.system(f"kubectl apply -f {PATH}/triton-single-{model_name}.yaml -n default")

def setup_linear_seldon_graph_transformer(
    model_name_one: str,
    model_name_two: str,
    model_uri_one: str,
    model_uri_two):
    pass

def setup_linear_seldon_graph_business(
    model_name_one: str,
    model_name_two: str,
    model_uri_one: str,
    model_uri_two):
    pass


setup_minio_secret(username="minioadmin", password="minioadmin")
setup_triton_single(
    model_name="resnet",
    model_uri="s3://separate-servers/separate-servers/resnet")
setup_triton_single(
    model_name="yolo",
    model_uri="s3://separate-servers/separate-servers/yolo")
