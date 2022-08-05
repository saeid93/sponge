import requests

payload = {}
response = requests.post(url='http://localhost:30800/v2/repository/models/xception/unload')


print(response)