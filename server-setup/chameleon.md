# Chameleon
Options
1. Using Microk8s
  1.1. Use the image or install Microk8s from its [documentation](https://microk8s.io/)
  1.2. Enable the GPU of the Microk8s [add-on:gpu](https://microk8s.io/docs/addon-gpu)
  1.3. stress test GPU stress [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  1.4. I have build the image for it with name microk8s-cluster in the Chameleon repository, just use it to fire up a server

2. Using Minikube NOT WORKING
  1.1. Use the image or install Minikube from its [documentation](https://minikube.sigs.k8s.io/docs/)
  1.2. Enable the GPU of the Minikube [NVIDIA GPU Support](https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/)
  1.3. stress test GPU stress [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  1.4. I have build the image for it with name minikube-cluster in the Chameleon repository, just use it to fire up a server

  resources to look [resource 1](https://anencore94.github.io/2020/08/19/minikube-gpu.html)

3. Using my bare metal single node cluster [TODO]
  1.1. 