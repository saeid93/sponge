import requests

payload = {}
response = requests.post(url='http://localhost:30800/v2/repository/models/resnet/versions/1/load', data=payload, allow_redirects=True)


print(response)