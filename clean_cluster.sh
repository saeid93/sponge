kubectl delete seldondeployment --all --force -n default
kubectl delete deployments --all --force -n default
kubectl delete replicaset --all --force -n default
kubectl delete pods --all --force -n default
kubectl get services | grep -v kubernetes | awk '{print $1}' | xargs kubectl delete service -n default