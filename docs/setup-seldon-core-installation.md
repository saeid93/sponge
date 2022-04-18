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
2. setup the ingress with istio [Guide-external](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html)
3. 



1. Install it easily using helm like the documentation from this notebook [seldon_core_setup.ipynb](seldon-core/notebooks/seldon_core_setup.ipynb)
2. Install the ingress for istio (not the ssl one)
3. Related notebooks
    1. [server_examples.ipynb](seldon-core/notebooks/server_examples.ipynb): build up a simple sklearn model from a pretrained model
    2. 

# -------------------

1. Setup ingress with [ambassodor](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/ambassador.html) (install the Ambassador API Gateway) or [istio](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html). Don't forget to port-forward in the case of abassodor or TODO in the case of istio
2. Install seldon core from [documentation](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html) and use this jupyter notebook [seldon_core_setup.ipynb](seldon_core_setup.ipynb)
3. Don't forget to port-forward like the end of the notebook [seldon_core_setup.ipynb](seldon_core_setup.ipynb)
```kubectl port-forward $(kubectl get pods -n seldon-system -l app.kubernetes.io/name=ambassador -o jsonpath='{.items[0].metadata.name}') -n seldon-system 8003:8080```
4. for abassodor installation
    1. To get the IP address of Ambassador, run the following commands:
        NOTE: It may take a few minutes for the LoadBalancer IP to be available.
        
        You can watch the status of by running

            kubectl get svc -w  --namespace seldon-system ambassador
            
        On GKE/Azure:
        
            export SERVICE_IP=$(kubectl get svc --namespace seldon-system ambassador -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

        On AWS:

            export SERVICE_IP=$(kubectl get svc --namespace seldon-system ambassador -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        
        Get the ip at:

            echo http://$SERVICE_IP:
    2. 