On the remote (load tester) machine steps starts with $

On the cluster main node starts with £

Setup ssh connection:

1. $ Generate node key using ssh-keygen
2. £ Copy the generated load tester machine key into the main node public keys of the cluster
3. £ Connect one time to the main machine on the load tester and prompt `yes` to the question
5. £ Add the IP of the main to the microk8s certificates as explained [here](./setup-chameleon-k8s.md)
6. $ Copy the main node kubeconfig to the remote cluster and also addd the IP of the remote to:
```bash
    server: https://192.5.86.160:16443
  name: microk8s-cluster
```
7. To sync two servers time refer to the [common problems](./common-problems.md)

8. We use the tc tool for limiting the bandwidth between two servers, follow [this guide for bandwidth limitation](https://netbeez.net/blog/how-to-use-the-linux-traffic-control/)
Examples:

```bash
# limitin the bandwidth for interfance ens3
sudo tc qdisc add dev ens3 root tbf rate 100kbit burst 32kbit latency 400ms

# showing limited bandwidth for interfance ens3
tc qdisc show  dev ens3
# example output:
# qdisc tbf 8001: root refcnt 2 rate 100Kbit burst 4Kb lat 400.0ms

# removing the limit on the bandwidth
sudo tc qdisc del dev ens3 root
```

**Note**: Since we use K8S [NodePort](https://kubernetes.io/docs/concepts/services-networking/service/#type-nodeport) method, using any of the machines IPs will work as a connection to the entire Clsuter
