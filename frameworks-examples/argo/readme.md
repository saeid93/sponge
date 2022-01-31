# Installation
[Full documentation (argocd)](https://argo-cd.readthedocs.io/en/stable/getting_started/)

```
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

[Full documentation (argo-workflow)](https://argoproj.github.io/argo-workflows/)

```
kubectl create ns argo
kubectl apply -n argo -f https://raw.githubusercontent.com/argoproj/argo-workflows/master/manifests/quick-start-postgres.yaml
```

# Accessing UI (easiest way)
Argocd
```
kubectl port-forward -n argocd svc/argocd-server 8080:443
```
Argo-workflow
```
kubectl -n argo port-forward deployment/argo-server 2746:2746
```
[Other methods (Load balander and Ingress) (argocd)](https://argo-cd.readthedocs.io/en/stable/getting_started/)
[Other methods (Load balander and Ingress) (argoworkflow)](https://argoproj.github.io/argo-workflows/argo-server/#access-the-argo-workflows-ui)

Gettring the sceret

```
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

# Simple app example
[Gitlab](https://gitlab.com/nanuchi/argocd-app-config)
[Video](https://youtu.be/MeU5_k9ssrs)

# Argo cli example
```
argo submit -n argo --watch https://raw.githubusercontent.com/argoproj/argo-workflows/master/examples/hello-world.yaml
argo list -n argo
argo get -n argo @latest
argo logs -n argo @latest
```