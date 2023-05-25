kubectl delete seldondeployment --all  --grace-period=10 -n default
kubectl delete deployments --all --grace-period=10 -n default
kubectl delete replicaset --all --grace-period=10 -n default
kubectl delete pods --all --grace-period=10 -n default
kubectl get services | grep -v kubernetes | awk '{print $1}' | xargs kubectl delete service -n default