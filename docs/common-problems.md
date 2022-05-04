# List of common problems

1. The server might suddenly stop working with this error
```
The connection to the server 127.0.0.1:16443 was refused - did you specify the right host or port?
```
I tried many things but the easiest is to re-install the Microk8s :( do it in purge mode so it will be deleted and re-installed fast
```
sudo snap remove --purge microk8s
sudo snap install microk8s --classic --channel=1.19/stable
```

2. Versioning of the packages should follow this in nodes:
[related issue](https://github.com/pallets/flask/issues/4494)
```
seldon-core
Flask==1.1.1
Jinja2==3.0.3
itsdangerous==2.0.1
Werkzeug==2.0.2
```

3. Sometimes namespaces stuck in the terminating state use the following command in that case [source](https://stackoverflow.com/questions/52369247/namespace-stuck-as-terminating-how-i-removed-it/63066925):
```
NAMESPACE=your-rogue-namespace
kubectl proxy &
kubectl get namespace $NAMESPACE -o json |jq '.spec = {"finalizers":[]}' >temp.json
curl -k -H "Content-Type: application/json" -X PUT --data-binary @temp.json 127.0.0.1:8001/api/v1/namespaces/$NAMESPACE/finalize
```
