# In-place vertical pod installation

according to [github](https://github.com/canonical/microk8s/issues/4355) we should use configs for microk8s


enable this feature flag `InPlacePodVerticalScaling=true` in the mirok8s config file

Do the following for installing Kubernetes with this feature:

```bash
sudo snap remove --purge microk8s
sudo mkdir -p /var/snap/microk8s/common/
sudo cp $HOME/hack/microk8s-config.yaml /var/snap/microk8s/common/.microk8s.yaml
sudo snap install microk8s --classic --channel 1.27
sudo microk8s config > $HOME/.kube/config
```