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
```
seldon-core
Flask==1.1.1
Jinja2==3.0.3
itsdangerous==2.0.1
Werkzeug==2.0.2
```