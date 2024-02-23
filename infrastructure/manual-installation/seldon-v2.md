# Install Seldon V2
Follow the mentioned steps to completely bring up the Seldon V2

## Uninstall old Seldon V2
```bash
helm uninstall seldon-v2-servers --namespace seldon-mesh

helm uninstall seldon-core-v2  --namespace seldon-mesh

helm uninstall seldon-core-v2-crds

kubectl delete namespace seldon-mesh
```

### DELETE namespace forcefully
```log
kubectl get namespace seldon-mesh -o json > tempfile.json

# change spec to be like this
# "spec" : {
#    }

kubectl replace --raw "/api/v1/namespaces/seldon-mesh/finalize" -f ./tempfile.json
```

## Installing Kafka (via Strimzi)
```bash
kubectl create namespace kafka
kubectl apply -f https://strimzi.io/install/latest?namespace=kafka -n kafka
kubectl apply -f https://strimzi.io/examples/latest/kafka/kafka-ephemeral.yaml -n kafka

# compare the results with the expected results
kubectl get pods -n kafka

NAME                                          READY   STATUS    RESTARTS   AGE
strimzi-cluster-operator-7f87b79897-mm82s     1/1     Running   0          8m43s
my-cluster-zookeeper-0                        1/1     Running   0          8m15s
my-cluster-zookeeper-1                        1/1     Running   0          8m15s
my-cluster-zookeeper-2                        1/1     Running   0          8m15s
my-cluster-kafka-0                            1/1     Running   0          7m51s
my-cluster-kafka-2                            1/1     Running   0          7m50s
my-cluster-kafka-1                            1/1     Running   0          7m51s
my-cluster-entity-operator-78fb9b9cb4-fj4nw   3/3     Running   0          7m23s
```


## Install Seldon
[Old Link (v2.4.0) -> not worked anymore](https://docs.seldon.io/projects/seldon-core/en/v2.4.0/contents/getting-started/kubernetes-installation/helm.html)

[New Link (v2.6.0) -> worked](https://docs.seldon.io/projects/seldon-core/en/v2.6.0/contents/getting-started/kubernetes-installation/helm.html)

```bash
microk8s enable dns storage observability

helm repo add seldon-charts https://seldonio.github.io/helm-charts
helm repo update seldon-charts

helm install seldon-core-v2-crds  seldon-charts/seldon-core-v2-crds

kubectl create namespace seldon-mesh

# use custom helm values (custom-seldon-core-v2-values.yaml) for installing seldon-core-v2 according to the kafka installation:
# set kafka.bootstrap to my-cluster-kafka-bootstrap.kafka:9092
# set opentelemetry.endpoint to prometheus-operated.observability:9090

cat custom-seldon-core-v2-values.yaml
```
```yaml
opentelemetry:
  endpoint: prometheus-operated.observability:9090
  enable: true
  ratio: 1

kafka:
  debug:
  bootstrap: my-cluster-kafka-bootstrap.kafka:9092
  topicPrefix: seldon
  consumer:
    sessionTimeoutMs: 6000
    autoOffsetReset: earliest
    topicMetadataRefreshIntervalMs: 1000
    topicMetadataPropagationMaxMs: 300000
    messageMaxBytes: 1000000000
  producer:
    lingerMs: 0
    messageMaxBytes: 1000000000
  topics:
    replicationFactor: 1
    numPartitions: 1
```
```bash
# install in one command (Recommended)
helm install seldon-core-v2 seldon-charts/seldon-core-v2-setup --namespace seldon-mesh --values custom-seldon-core-v2-values.yaml

# install in two commands
helm install seldon-core-v2 seldon-charts/seldon-core-v2-setup --namespace seldon-mesh
helm upgrade seldon-core-v2 seldon-charts/seldon-core-v2-setup --values custom-seldon-core-v2-values.yaml -n seldon-mesh

# added in seldon v2.6.0 (not included in v2.4.0)
helm install seldon-v2-runtime seldon-charts/seldon-core-v2-runtime --namespace seldon-mesh

helm install seldon-v2-servers seldon-charts/seldon-core-v2-servers --namespace seldon-mesh

# match the pods with the expected ones
kubectl get pods -n seldon-mesh

NAME                                         READY   STATUS    RESTARTS   AGE
seldon-controller-manager-58cb6cdf86-s5b9r   1/1     Running   0          9m50s
seldon-pipelinegateway-86458ff8b5-7mdgv      1/1     Running   0          9m38s
seldon-modelgateway-7d8d78dbbf-kzsnq         1/1     Running   0          9m38s
hodometer-6946f6d879-9zs8j                   1/1     Running   0          9m38s
seldon-envoy-7957bfcd6d-lcf69                1/1     Running   0          9m38s
seldon-scheduler-0                           1/1     Running   0          9m38s
triton-0                                     3/3     Running   0          9m8s
seldon-dataflow-engine-5974f7dcf9-zl68w      1/1     Running   0          55s
mlserver-0                                   3/3     Running   0          52s
```

## Troubleshoot
### Problem: You are getting errors like the following:
```log
ERROR   Reconciler error        {"controller": "server", "controllerGroup": "mlops.seldon.io", "controllerKind": "Server", "server": {"name":"triton","namespace":"seldon-mesh"}, "namespace": "seldon-mesh", "name": "triton", "reconcileID": "d124e7a7-c62d-4abc-82bc-3593c1b57c88", "error": "rpc error: code = Unavailable desc = connection error: desc = \"transport: Error while dialing dial tcp: lookup seldon-scheduler.seldon-mesh on 10.152.183.10:53: no such host\""}
```
### Solution: uninstall the seldon completey (untill deleting namespace) and install the last version which tagged by semver

### Solution2: If the problem has not been resolved, try to enable `ingress` with `microk8s enable ingress` and then completely uninstall the seldon and its namespace and install it again.
