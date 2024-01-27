import requests

model_name = "yolo"
response = requests.post(
    # f"http://localhost:32000/seldon/default/{model_name}/v2/repository/models/{model_name}/load",
    "http://localhost:8080/v2/repository/models/yolo/load",
    json={"max_batch_size": 16},
)
print(response)
