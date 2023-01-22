# Nvidia gpu

First of all, enable gpu:

```
microk8s enable gpu
```

 then, activate nvidia on your system with kubectl

```
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

```

after that, build time-slicing-config.yaml like this:

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator-resources
data:
    a100-40gb: |-
        version: v1
        sharing:
          timeSlicing:
            resources:
            - name: nvidia.com/gpu
              replicas: 8
            - name: nvidia.com/mig-1g.5gb
              replicas: 2
            - name: nvidia.com/mig-2g.10gb
              replicas: 2
            - name: nvidia.com/mig-3g.20gb
              replicas: 3
            - name: nvidia.com/mig-7g.40gb
              replicas: 7
    tesla-t4: |-
        version: v1
        sharing:
          timeSlicing:
            resources:
            - name: nvidia.com/gpu
              replicas: 10

```
change 10 to your desire number.

```
kubectl label node <node-name> nvidia.com/device-plugin.config=tesla-t4 
```

after that:

```
kubectl create -f time-slicing-config.yaml
```

```
kubectl patch clusterpolicy/cluster-policy \
   -n gpu-operator-resources --type merge \
   -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config"}}}}'
```

then check the node to make sure for the result.

```
kubectl describe <node>
```