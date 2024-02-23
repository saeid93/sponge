#!/bin/bash

install_kubectl() {
    echo "Install kubectl"
    curl -LO https://dl.k8s.io/release/v1.23.2/bin/linux/amd64/kubectl
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    sudo microk8s config > $HOME/.kube/config
    sudo ufw allow 16443
    echo "y" | sudo ufw enable
    echo "End Install kubectl"
    rm kubectl
    echo "alias k='kubectl'" >> ~/.zshrc
}

connect_to_remote() {
    mkdir ~/.kube
    REMOTE_USER="cc"
    REMOTE_KUBECONFIG="~/.kube/config"
    LOCAL_KUBECONFIG=~/.kube/config
    echo  scp $REMOTE_USER@$REMOTE_IP:$REMOTE_KUBECONFIG $LOCAL_KUBECONFIG
    scp $REMOTE_USER@$REMOTE_IP:$REMOTE_KUBECONFIG $LOCAL_KUBECONFIG
    sed -i "s/https:\/\/[0-9\.]\+:16443/https:\/\/$REMOTE_IP:16443/g" $LOCAL_KUBECONFIG
    echo "kubeconfig file copied from remote machine to local machine and server IP updated."
}

install_kubectl
connect_to_remote
