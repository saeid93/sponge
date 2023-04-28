kubectl delete seldondeployment --all -n istio-system
kubectl delete deployments --all -n istio-system
kubectl delete replicaset --all -n istio-system
kubectl delete pods --all -n istio-system
kubectl get services | grep -v kubernetes | awk '{print $1}' | xargs kubectl delete service -n istio-system