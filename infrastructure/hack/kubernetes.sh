#!/bin/bash

# Install Helm
function install_helm() {
    echo "Installing Helm"
    wget https://get.helm.sh/helm-v3.11.3-linux-amd64.tar.gz -O helm.tar.gz
    tar -xf helm.tar.gz
    sudo mv linux-amd64/helm /usr/local/bin/helm
    rm helm.tar.gz
    rm -r linux-amd64
    echo "Helm installation complete"
    echo
}

# Install MicroK8s
function install_microk8s() {
    if [ "$VPABRANCH" = "Yes" ]; then
        sudo mkdir -p /var/snap/microk8s/common/
        sudo cp $HOME/infrastructure/hack/microk8s-config.yaml /var/snap/microk8s/common/.microk8s.yaml
    fi
    echo "Installing MicroK8s"
    sudo snap install microk8s --classic --channel=$K8SVERSION/edge
    sudo usermod -a -G microk8s cc
    mkdir -p $HOME/.kube
    sudo chown -f -R cc ~/.kube
    microk8s config > $HOME/.kube/config
    sudo ufw allow in on cni0
    sudo ufw allow out on cni0
    sudo ufw default allow routed
    sudo microk8s enable dns
    if [ "$GPU" = "Yes" ]; then
        microk8s enable gpu
    fi
    echo "alias k='kubectl'" >> ~/.zshrc
    echo "MicroK8s installation complete"
    echo
}

# Install Minikube
function install_minikube() {
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube
    rm minikube-linux-amd64
    if [ "$VPABRANCH" = "Yes" ]; then
        minikube start --kubernetes-version=v1.27.0 --feature-gates=InPlacePodVerticalScaling=true
    else
        minikube start
    fi
    echo "Installing Minikube"
    echo "alias k='kubectl'" >> ~/.zshrc
    echo "Minikube installation complete"
    echo
}


echo "Running kubernetes script"
install_helm

if [ "$VPABRANCH" = "Yes" ]; then
    install_minikube
else
    install_microk8s
fi

echo "Kubernetes script execution complete"
