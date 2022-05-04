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
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: 10.140.81.236
    path: "/mnt/myshareddir"
EOF
```
2. Associate a PVC to the generated PV
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-nfs
  namespace: default
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
EOF
```
Bare in mind if you delete a PV or PVC you first delete the associated resource with that PVC first, otherwise it will stuck at the terminating state.




## Resources
[Persistent Volume, Persistent Volume Claim & Storage Clas- Nana](https://youtu.be/0swOh5C3OVM)
[NFS Persistent Volume in Kubernetes Cluster - justmeandopensource](https://youtu.be/to14wmNmRCI)
