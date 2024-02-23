First of all, make sure you microk8s version is eqeual to fucking 22/stable version.
```
  microk8s enable gpu
```
Go with instruction suggests in [nvidia](https://github.com/NVIDIA/k8s-device-plugin)
make sure your KUBECONFIG is client.config which microk8s provided. For setting it:

```
  export KUBECONFIG=/var/snap/microk8s/current/.../client.conig
```

Look at ``` k describe node <Node Name> ```
If you see nvidia.com/gpu in capacity part, you are in a right place; else cry and search(In the time I am writing this readme, I really do not know what was my fucking problems precisely).

