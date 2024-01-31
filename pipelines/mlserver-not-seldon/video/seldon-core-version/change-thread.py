import requests

model_name = "yolo"
response = requests.post(
    f"http://localhost:32000/seldon/default/{model_name}/change",
    # "http://localhost:8080/change",
    json={
        "interop_threads": 4,
        "num_threads": 4
    },
)
print(response)
