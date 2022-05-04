## Installing NFS
NFS is our storage file system backend for more information on K8S storage system see [Volummes-k8s-doc](https://kubernetes.io/docs/concepts/storage/volumes/)

1. Follow the instructions [installing-nfs](https://cloud.netapp.com/blog/azure-anf-blg-linux-nfs-server-how-to-set-up-server-and-client)
2. In the export file find the Chameleon IP network and mask and add it to the export file, to find the network IP and mask:
`
sudo ifconfig | grep -i mask
`
3. Install both client and server folders on your machines and make sure that a file created on one of them is accessible on the client folder too.


## Setup up Persistent Volume and Persistent Volume Claim
In k8s you have to set a PV and a claim for that PVC attached to your volume type which in our case is NFS
1. Set up the PV and point it to the NFS directory and IP (in our case 10.140.81.236 and /mnt/myshareddir)
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
  namespace: default
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: 10.140.81.236
    path: "/mnt/myshareddir"
EOF
```
2. Associate a PVC to the generated PV
```
kubectl create ns minio
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-nfs
  namespace: minio
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
EOF
```
Bare in mind if you delete a PV or PVC you first delete the associated resource with that PVC first, otherwise it will stuck at the terminating state.

## Minio installation
Install the Minio with helm and set the value to our existing pvc
```
helm repo add minio https://helm.min.io/
helm install --namespace minio \
--set rootUser=minioadmin,rootPassword=minioadmin \
--set persistence.existingClaim=pvc-nfs \
--generate-name minio/minio 
```

4. continue the rest of steps from the printed instructions\
5. find release name from `helm list` and then:
To access Minio from localhost, run the below commands:
```
export POD_NAME=$(kubectl get pods --namespace minio -l "release=<release-name>" -o jsonpath="{.items[0].metadata.name}")
kubectl port-forward $POD_NAME 9000 --namespace minio
```
Read more about port forwarding here: http://kubernetes.io/docs/user-guide/kubectl/kubectl_port-forward/
You can now access Minio server on http://localhost:9000. Follow the below steps to connect to Minio server with mc client:
Download the Minio mc client - https://docs.minio.io/docs/minio-client-quickstart-guide Get the
```
ACCESS_KEY=$(kubectl get secret minio-1651658304 -o jsonpath="{.data.accesskey}" | base64 --decode) and the SECRET_KEY=$(kubectl get secret minio-1651658304 -o jsonpath="{.data.secretkey}" | base64 --decode)
mc alias set minio-1651658304-local http://localhost:9000 "$ACCESS_KEY" "$SECRET_KEY" --api s3v4
mc ls minio-1651658304-local
```
Alternately, you can use your browser or the Minio SDK to access the server - https://docs.minio.io/categories/17

## Resources
[Persistent Volume, Persistent Volume Claim & Storage Clas- Nana](https://youtu.be/0swOh5C3OVM) \
[NFS Persistent Volume in Kubernetes Cluster - justmeandopensource](https://youtu.be/to14wmNmRCI)
