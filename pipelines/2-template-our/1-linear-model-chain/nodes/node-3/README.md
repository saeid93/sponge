A simple microservice that doubles the input


For attaching to bash of the model:
```
docker run -it linearmodel:nodethree bash
```
For testing the model endpoints on docker
```
docker run --rm -p 5001:5000 linearmodel:nodethree
```
In the terminal:
```
curl -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' -X POST http://localhost:5001/predict -H "Content-Type: application/json"
```