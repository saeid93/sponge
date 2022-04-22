# Microk8s
1. Istio is available as one of the [Microk8s addons](https://istio.io/latest/docs/setup/platform-setup/microk8s/), simply enable it as:
```
microk8s enable istio
```
2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample application

3. To enable Prometheus and Grafana for Istio (for latency monitoring)
Follow [Visualizing Metrics with Grafana
](https://istio.io/latest/docs/tasks/observability/metrics/using-istio-dashboard/) and try the book-info app to make sure it is up and running. TODO not tested


4. To enable Kiali (for monitoring traces)
TODO

5. To enable Jeager
TODO


# Minikube
1. Install it using [istio-documentation-install-minikube](https://istio.io/latest/docs/setup/platform-setup/minikube/)
2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample application

3. To enable Prometheus and Grafana for Istio
TODO

4. To enable Kiali
TODO

5. To enable Jeager
TODO

# Bare metal
1. Download and install using the [official-istio-documentation](https://istio.io/latest/docs/setup/getting-started/)
2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample application

3. To enable Prometheus and Grafana for Istio
TODO

4. To enable Kiali
TODO

5. To enable Jeager
TODO
