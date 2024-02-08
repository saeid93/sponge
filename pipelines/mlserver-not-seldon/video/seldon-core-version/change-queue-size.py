# from kubernetes import client, config
# import json
# import base64

# # Load the Kubernetes configuration from the default location
# config.load_kube_config()

# # Create a Kubernetes API client
# api_client = client.CoreV1Api()

# # Get the name of the Pod containing the container you want to modify
# pod_name = "my-pod"

# # Get the name of the container you want to modify
# container_name = "my-container"

# # Get the current environment variables of the container
# container = api_client.read_namespaced_pod(pod_name, namespace="default")
# env_vars = container.spec.containers[0].env

# # Modify the value of the environment variable you want to change
# for env_var in env_vars:
#     if env_var.name == "MLSERVER_MODEL_MAX_BATCH_SIZE":
#         env_var.value = "new-value"

# # Encode the modified environment variables as a JSON string
# env_json = json.dumps([{"name": env_var.name, "value": env_var.value} for env_var in env_vars])
# env_base64 = base64.b64encode(env_json.encode('utf-8')).decode('utf-8')

# # Patch the container's environment variables using kubectl exec
# command = f"export MLSERVER_MODEL_MAX_BATCH_SIZE={env_base64}; envsubst < /dev/null"
# exec_command = [
#     "/bin/sh",
#     "-c",
#     command
# ]
# api_client.connect_get_namespaced_pod_exec(
#     pod_name,
#     "default",
#     command=exec_command,
#     container=container_name,
#     stdin=True,
#     stdout=True,
#     stderr=True,
#     tty=False,
# )


import requests


# model_name = "queue-resnet-human"
model_name = "queue-yolo"
response = requests.post(
    f"http://localhost:32000/seldon/default/{model_name}/v2/repository/models/{model_name}/load",
    json={"max_batch_size": 16}
)
print(response)
