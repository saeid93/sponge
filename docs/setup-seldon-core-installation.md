# Seldon core isntallation - Istio

Seldon is a framework for making complex grpc and rest apis for the trained ML models

1. According to [doc](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html) install the istio version
```
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
```
2. setup the ingress with istio (prefarably with tmux and detach the window so you don't have set this up everytime) [Guide](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html)
```
cat <<EOF | kubectl apply -f -          
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF
```

The `metadata.name` field is how the Seldon core is able to detect this Gateway. Bear in mind that the Gateway should also be in the `istio-system` namespace.

To access Seldon Services you have three options:

3.2. **Node Port (Recommended)** Edit the `kubectl edit service istio-ingressgateway -n istio-system` and make this change
```
  - name: http2
    nodePort: 32000
    port: 80
    protocol: TCP
    targetPort: 8080
```
The Seldon core is available on `<cluster-ip>:32000`

3.1. **Port Forward** to the ingress port 80 (since the isio ingress you deployd in the former step is operating on port 80) to port 8004 and detach tmux (therfore the connection will stay open) - if you are using microk8s istio use the port 8080 as the target-port for http is 8080 in microk8s istio.
```
kubectl port-forward $(kubectl get pods -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].metadata.name}') -n istio-system 8004:8080
```
The Seldon core is ready to go on localhost:8004! For information about the Seldon core endpoint addresses see [endpoint-references](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html#istio-configuration-annotation-reference)


3.3. **Ingress**


