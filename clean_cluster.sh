kubectl delete seldondeployment --all -n default
kubectl delete deployments --all -n default
kubectl delete replicaset --all -n default
kubectl delete pods --all -n default
kubectl get services | grep -v kubernetes | awk '{print $1}' | xargs kubectl delete service -n default