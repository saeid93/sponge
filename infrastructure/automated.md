# Inference Project Automation

1. `make zsh` to install zshell; reply 'Y' to setting zsh as the default

2. For main cluster installation `make all PROJECT=<ipa or inference_x or malleable-scaler or all> K8SVERSION=<kubernetes version> PUBLIC_IP=<server's public IP> GPU=<Yes or No> VPABRANCH=<Yes or No>` to do the rest of the installation steps.

3. To make sure IPA routers are always scheduled on the manager node:
```
kubectl label node <manager node name> router-node=true
```
You can get the manager node name with:
```
hostname
```

4. For load_tester installation `make load_tester PROJECT=<ipa or inference_x or all>` to do the rest of the installation steps.

5. For external load tester access (create the load tester machine after the main cluster). On the remote (load tester) machine steps starts with $, on the cluster main node starts with £:
    1. $ On the load tester node generate and show the public key:
    ```bash
    make generate_and_show_key
    ```
    3. £ On the main node of the cluster save the load tester machine public key in the `~/infrastructure/hack/load-tester-key` and then, enable access (ssh and kubectl) from the remote machin to the load_tester on the main node:
    ```bash
    make enable_ssh_and_kubectl
    ```
    3. $ On the load tester do the following:
    ```bash
    make remote_kubectl_install REMOTE_IP=<main node of K8S cluster IP>
    ```
    4. To sync time between servers
        1. £ On main node:
        ```bash
        make sync_time_main
        ```
        2. $ On load tester:
        ```bash
        make sync_time_load_tester PRIMARY_IP=<ip of the main node>
        ```
