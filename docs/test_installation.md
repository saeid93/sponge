To test Seldon core endpoint correct configuration and installation deploy the following:
```
cat <<EOF | kubectl apply -f -
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  predictors:
  - graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.14.0-dev/sklearn/iris
    name: default
    replicas: 1
    svcOrchSpec:
      env:
      - name: SELDON_LOG_LEVEL
        value: DEBUG
EOF
```

On your cluster nodes:
```
curl -s -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' \
-X POST http://localhost:32000/seldon/default/sklearn/api/v1.0/predictions -H \
"Content-Type: application/json"
```
and on outside cluster machine:
```
CLUSTER_NODE_IP=192.5.86.160
curl -s -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' \
-X POST http://$CLUSTER_NODE_IP:32000/seldon/default/sklearn/api/v1.0/predictions -H \
"Content-Type: application/json"
```
In both cases the result should be:
```
{"data":{"names":["t:0","t:1","t:2"],"ndarray":[[9.912315378486697e-07,0.0007015931307746079,0.9992974156376876]]},"meta":{"requestPath":{"classifier":"seldonio/sklearnserver:1.13.1"}}}
```
