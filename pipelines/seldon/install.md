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