# Seldon core isntallation - Istio

Seldon is a framework for making complex grpc and rest apis for the trained ML models

1. According to [doc](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html) install the istio version
```
kubectl create namespace seldon-system \
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
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF
```
port forward to the ingress port 80 (since the isio ingress you deployd in the former step is operating on port 80) to port 8004 and detach tmux (therfore the connection will stay open) - ! BUG sometimes you have to port forward to 8080 instead of 80 for the reason that is yet unknown to me.
```
kubectl port-forward $(kubectl get pods -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].metadata.name}') -n istio-system 8004:8080
```
4. The Seldon core is ready to go on port 8004! For information about the Seldon core endpoint addresses see [endpoint-references](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html#istio-configuration-annotation-reference)


