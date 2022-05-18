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
4. metrics-server might suddenly stops working with the following output in the logs
```
panic: failed to create listener: failed to listen on 0.0.0.0:443: listen tcp 0.0.0.0:443: bind: permission
```
edit the metrics-server deployment:
```
kubectl edit deployment metrics-server -n kube-system
```
and change the port from 443 to 4443:
```
serviceAccountName: metrics-server
volumes:
# mount in tmp so we can safely use from-scratch images and/or read-only containers
- name: tmp-dir
  emptyDir: {}
containers:
- name: metrics-server
  image: k8s.gcr.io/metrics-server-amd64:v0.3.6
  args:
    - --cert-dir=/tmp
    - --secure-port=4443
  ports:
  - name: main-port
    containerPort: 4443
    protocol: TCP
  securityContext:
    readOnlyRootFilesystem: true
    runAsNonRoot: true
    runAsUser: 1000
  imagePullPolicy: Always
  volumeMounts:
```

5. Following https://github.com/SeldonIO/seldon-core/issues/4102 avoid calling load on the server's init function.
