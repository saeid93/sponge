# Microk8s
1. Istio is available as one of the [Microk8s addons](https://istio.io/latest/docs/setup/platform-setup/microk8s/), simply enable it as:
```
microk8s enable istio
```
2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample applicatio

3. To install Kiali (for monitoring traces)
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
```
4. to enable outside access to it change it service to nodeport on port 32001
```
kubectl edit svc kiali -n istio-system
```


# Minikube
1. Install it using [istio-documentation-install-minikube](https://istio.io/latest/docs/setup/platform-setup/minikube/)

2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample applicatio

3. To install Kiali (for monitoring traces)
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
```
4. to enable outside access to it change it service to nodeport on port 32001
```
kubectl edit svc kiali -n istio-system
```

5. To enable Jeager
TODO

# Bare metal
1. Download and install using the [official-istio-documentation](https://istio.io/latest/docs/setup/getting-started/)

2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample applicatio

3. To install Kiali (for monitoring traces)
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
```
4. to enable outside access to it change it service to nodeport on port 32001
```
kubectl edit svc kiali -n istio-system
```
