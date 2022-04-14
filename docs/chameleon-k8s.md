# Options for Installing K8S
Here are the options for installing K8S on Chameleon cloud with GPU Support
* **Using Microk8s**
  1. Use the image or install Microk8s from its [documentation](https://microk8s.io/)
  2. Enable the GPU of the Microk8s [add-on:gpu](https://microk8s.io/docs/addon-gpu)
  3. stress test GPU for checking activeness [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  4. I have build the image for it with name microk8s-cluster in the Chameleon repository, just use it to fire up a server

* **Using Minikube** NOT WORKING
  1. Use the image or install Minikube from its [documentation](https://minikube.sigs.k8s.io/docs/)
  2. Enable the GPU of the Minikube [NVIDIA GPU Support](https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/)
  3. stress test GPU for checking activeness [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  4. I have build the image for it with name minikube-cluster in the Chameleon repository, just use it to fire up a server
  5. External Resources
      * [resource 1](https://anencore94.github.io/2020/08/19/minikube-gpu.html)

* **Using bare metal cluster** TODO
  1. TODO