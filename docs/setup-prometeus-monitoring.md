# Microk8s

## kubernetes
1. following [doc](https://microk8s.io/docs/addon-dashboard) enable the dashboard
```
microk8s enable dashboard
```
2. Get the token
```
token=$(microk8s kubectl -n kube-system get secret | grep default-token | cut -d " " -f1)
microk8s kubectl -n kube-system describe secret $token
```
3. Access the dashboard in the following address:

http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/


## Prometheus and Grafana

1. Just use the builtin add-on
```
microk8s enable prometheus
```
2. For enabling outside access make the `service/prometheus-k8s` and `service/grafana` of type NodePort isntead of ClusterIP using the following command and editing the `type` field
```
kubectl edit svc prometheus-k8s -n monitoring
kubectl edit svc grafana -n monitoring
```
3. Both of the grafana and prometheus are now accessible via the following links
```
<your node ip>:<prometheus port, defualt 31799>
<your node ip>:<grafana port, defualt 30460>
```

4. To integrate the Seldon core metrics into prometheus and grafana


5. To integrate Istio into prometheus and Grafana


# Minikube



# Bare Metal
TODO polish

We'll be installing this [this](docs/installing-prometheus.md) prometheus+grafana monitoring using Helm charts ([source](https://github.com/geerlingguy/kubernetes-101/tree/master/episode-10)). You can find guide to installing helm itself [here](https://helm.sh/docs/intro/install/). Alternate Prometeus installation solution [here](https://github.com/prometheus-operator/prometheus-operator).

This Helm chart installs the following in your cluster:

  - kube-state-metrics (gathers metrics from cluster resources)
  - Prometheus Node Exporter (gathers metrics from Kubernetes nodes)
  - Grafana
  - Grafana dashboards and Prometheus rules for Kubernetes monitoring

To install it, first add the Prometheus Community Helm repo and run `helm repo update`:

```
$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
$ helm repo update
```

Then install the stack into the `monitoring` namespace:

```
$ kubectl create namespace monitoring
$ helm install prometheus --namespace monitoring prometheus-community/kube-prometheus-stack
```

Watch the progress in Lens, or via `kubectl`:

```
$ kubectl get deployments -n monitoring -w
```

Once deployed, you can access Grafana using the default `admin` account and the default password `prom-operator`.


To access Grafana in your browser, run:

```
$ kubectl port-forward -n monitoring service/prometheus-grafana 3000:80
```

Then open your browser and visit `http://localhost/` and log in with the password you found from the earlier command.

To access prometheus in your browser, run:

```
$ kubectl port-forward -n monitoring service/prometheus-kube-prometheus-prometheus 9090:9090
```

Then open your browser and visit `http://localhost:9090/` and log in with the password you found from the earlier command.


## Unistalling Monitoring Stack

```
helm uninstall prometheus --namespace monitoring
```

This removes all the Kubernetes components associated with the chart and deletes the release.


CRDs created by this chart are not removed by default and should be manually cleaned up:
```
kubectl delete crd alertmanagerconfigs.monitoring.coreos.com
kubectl delete crd alertmanagers.monitoring.coreos.com
kubectl delete crd podmonitors.monitoring.coreos.com
kubectl delete crd probes.monitoring.coreos.com
kubectl delete crd prometheuses.monitoring.coreos.com
kubectl delete crd prometheusrules.monitoring.coreos.com
kubectl delete crd servicemonitors.monitoring.coreos.com
kubectl delete crd thanosrulers.monitoring.coreos.com
```

## Related promql commands
### contaienrs metrics

All containers memory usages:
```
container_memory_usage_bytes{container='sample-vpa'}
```
All containers cpu usages:
```
TODO
```
A specific container memory usage
```
TODO
```
A specific container cpu usage
```
TODO
```
### pods metrics

## Guide to Grafana dashboard