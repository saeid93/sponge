# Seldon core isntallation - Istio

Seldon is a framework for making complex grpc and rest apis for the trained ML models

1. According to [doc](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html) install the istio version
```
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
```
2. setup the ingress with istio [Guide](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html)
3. The Seldon core is ready to go!
4. For and an example app running see [istio-canary-example-notebook](../seldon-core-examples/istio/canary/istio_canary.ipynb) (remember to do the port forwarding in a terminal as instructed)
