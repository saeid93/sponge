curl command result in this not working
```
curl -v -s -d '{"meta": {"tags": {}},"data": {"names": ["message"],"ndarray": ["In an attets skills"]}}' -X POST http://localhost:8004/seldon/seldon/seldon-pipeline/api/v0.1/predictions -H 'Content-Type: application/json'
```
{"status":{"code":-1,"info":"Empty json parameter in data","reason":"MICROSERVICE_BAD_DATA","status":1}}
1. Tried to add the following depandancy - didn't work
```
Flask==1.1.1
Jinja2==3.0.3
itsdangerous==2.0.1
Werkzeug==2.0.2
```
2. Tried to change the the base seldon image