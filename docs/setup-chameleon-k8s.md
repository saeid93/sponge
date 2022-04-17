# Options for Installing K8S
Here are the options for installing K8S on Chameleon cloud with GPU Support
* **Microk8s**
  1. Use the image or install Microk8s from its [documentation](https://microk8s.io/)
  2. Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) for api access both on your local and your server
  3. To enable outside access to your cluster with kubectl in one of the cluster machines [get the kubecinfg](https://microk8s.io/docs/working-with-kubectl) and copy it to your server `~/.kube/config` file
  4. To enable outside externally (from an external machine e.g. your local) do the following:
      1. According to this [issue](https://github.com/canonical/microk8s/issues/421) on your cluster master node disable the firewall on the port 16443 which is the default apiserver port
      ```
      sudo ufw allow 16443
      sudo ufw enable
      ``` 
      2. Find out your master node public ip from the Chameleon dashboard and according to [question](https://stackoverflow.com/questions/63451290/microk8s-devops-unable-to-connect-to-the-server-x509-certificate-is-valid-f) use the instruction given in [authentication and authorization](https://stackoverflow.com/questions/63451290/microk8s-devops-unable-to-connect-to-the-server-x509-certificate-is-valid-f) to include the public ip in the server certificates. As said in this [answer](https://stackoverflow.com/a/65571967) that the ip should be `USE IP > 100`
  5. Enable the GPU of the Microk8s [add-on:gpu](https://microk8s.io/docs/addon-gpu)
  6. Stress test GPU for checking activeness [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  7. To add other nodes to the cluster TODO
  8. I have build the image for it with name microk8s-cluster in the Chameleon repository, just use it to fire up a server.

* **Minikube** NOT WORKING
  1. Use the image or install Minikube from its [documentation](https://minikube.sigs.k8s.io/docs/)
  2. Enable the GPU of the Minikube [NVIDIA GPU Support](https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/)
  3. stress test GPU for checking activeness [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  4. I have build the image for it with name minikube-cluster in the Chameleon repository, just use it to fire up a server
  5. External Resources
      * [resource 1](https://anencore94.github.io/2020/08/19/minikube-gpu.html)

* **Bare metal* TODO
  1. TODO
