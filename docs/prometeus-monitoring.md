## Installing Monitoring Stack

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