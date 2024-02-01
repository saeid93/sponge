import requests

model_name = "resnet-car"
response = requests.post(
    # f"http://localhost:32000/seldon/default/{model_name}/change",
    f"http://localhost:32002/change",
    # "http://localhost:8080/change",
    json={
        "interop_threads": 4,
        "num_threads": 4
    },
)
print(response)
