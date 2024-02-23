# List of common problems

1. The server might suddenly stop working with this error
```
The connection to the server 127.0.0.1:16443 was refused - did you specify the right host or port?
```
I tried many things but the easiest is to re-install the Microk8s :( do it in purge mode so it will be deleted and re-installed fast
```
sudo snap remove --purge microk8s
sudo snap install microk8s --classic --channel=1.24/stable
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
6. There is a known bug in Chameleon cloud, the mounted object storge keeps get disconnected, while fixing it these two command are simple workaround
```
sudo umount -l ~/my_mounting_point 
cc-cloudfuse mount ~/my_mounting_point
```
7. Server time being out of sync with the current UTC time (prometheus dashboard warns this). to fix, sync time with NTP server
```shell
sudo apt install systemd-timesyncd
sudo timedatectl set-ntp 1
sudo systemctl restart systemd-timedated.service
```
[comment]: # (sudo timedatectl set-time HH:MM:SS)
anotehr solution is using ntp:
```bash
sudo ufw allow 123/tcp
sudo ufw allow 123/utp
sudo apt update
sudo apt install ntp
sudo systemctl restart ntp
```

8. Access to apiserver impossible from remote hosts
    - 8.1. `sudo ufw allow 16443`
    - 8.2. Edit file /var/snap/microk8s/current/certs/csr.conf.template, Add the floating ip as IP.x
    - 8.3. mv /var/snap/microk8s/current/var/lock/no-cert-reissue /var/snap/microk8s/current/var/lock/no-cert-reissue_temp
    - 8.4. microk8s stop
    - 8.5. microk8s start
    - 8.6. mv /var/snap/microk8s/current/var/lock/no-cert-reissue_temp /var/snap/microk8s/current/var/lock/no-cert-reissue
9. GPU not detectable by Kubernetes:
    - https://github.com/canonical/microk8s/issues/448#issuecomment-623046044
10. For using huggigngface speech recognition use the following dataset:
    - Install `pip install "datasets[audio]"` (make sure "datasets[audio]" is in double quotation - This is fucking stupid but that's just how it is)
    - Follwoing [StackOverflow issue](https://stackoverflow.com/questions/38480029/libsndfile-so-1-cannot-open-shared-object-file-no-such-file-or-directory) isntall `sudo apt-get install libsndfile1-dev`
    - For the following problem do this `pip install -U torch torchaudio --no-cache-dir` according to [this issue](https://github.com/pytorch/audio/issues/62#issuecomment-1166196925)
```
RuntimeError: Failed to import transformers.models.speech_to_text.feature_extraction_speech_to_text because of the following error (look up to see its traceback):
/home/cc/miniconda3/envs/central/lib/python3.9/site-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK3c107SymBool10guard_boolEPKcl
```
11. For connecting a node to the cluster the hostname should be added. e.g.:
   - Go to the `sudo vim /etc/hosts` and add it
   - ```# kubernetes
        10.140.82.175 k8s-cluster
        10.140.81.149 k8s-cluster-gpu
      ```
12. Sometimes istio sidecar is not getting created and all the istio system is up and working perfectly. This is due to the reaseon that namespace of the pod is not labeled properly. According to [this](https://stackoverflow.com/questions/59734184/istio-side-car-is-not-created) do the following:
```
kubectl label namespace default istio-injection=enabled --overwrite
```
to check whether is has been properly labeled use the following command:
```
kubectl get namespace -L istio-injection
```
and see that your namespace is labeled:
```
NAME              STATUS   AGE   ISTIO-INJECTION
default           Active   26d   enabled
ingress           Active   18d   
istio-system      Active   18d   disabled
kube-node-lease   Active   26d   
kube-public       Active   26d   
```
13. For video monitoring pipeline we should use the following commands for opencv to work (according the [this](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)):
```
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```
14. Kubernetes nodes will go under disk pressure from time to time and the pods get stuck in the `pending` state, use the following command to flush all dangling and none-dangling images from the server
```
docker system prune -a
```
15. Don't forget to sync versioning of mlserver in your build files and re-installing `mlserver`
16. Chameleon DNS is unreliable, it not working following this [StackOverflow](https://stackoverflow.com/questions/52815784/python-pip-raising-newconnectionerror-while-installing-libraries) add `8.8.8.8` to your DNS:
```
nameserver 8.8.8.8 to the /etc/resolv.conf file. 8.8.8.8
```
17. MLServer is located at in the containers:
```
/opt/conda/lib/python3.8/site-packages/mlserver
```
18. Sometimes the server performance strangly drop. Reinstall the server and re-install all containers.
19. Sometimes the pytests command stops working with `pytest ...` and it only works with `python -m -pytest ...` a workaround to this is to add the folder to the `PYTHONPATH` manually: `export PYTHONPATH=$PYTHONPATH:.`, for vscdoe do the following steps to add somthing to `PYTHONPATH`:
Open your project in Visual Studio Code.
Press `Ctrl +` , to open the settings.
Click on the "Open Settings (JSON)" icon in the top-right corner of the Settings tab. This will open the settings.json file.
In the settings.json file, you can add the "python.envFile" setting to specify a file containing environment variables. Create a file (e.g., .env) in your project's root directory and add the `PYTHONPATH` variable. For example:
```
json
Copy code
{
  "python.envFile": "${workspaceFolder}/.env"
}
```
In the .env file, set the PYTHONPATH variable to the desired value. For example:
```
PYTHONPATH=/path/to/your/custom/modules
Replace /path/to/your/custom/modules with the actual path to the directory containing your custom Python modules.
```
Save the changes to the .env file.
Close and reopen Visual Studio Code for the changes to take effect.
