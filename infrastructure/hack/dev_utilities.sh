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
}

function install_docker() {
    echo "Install Docker"
    sudo apt-get remove -y docker docker-engine docker.io containerd runc
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo groupadd docker
    sudo usermod -aG docker $USER
    sudo systemctl enable docker.service
    sudo systemctl enable containerd.service
    rm get-docker.sh
    echo "End Install Docker"
    echo
}

# Install and activate Conda environment
function install_conda_environment() {
    echo "Removing Conda environment"
    conda deactivate
    conda env remove --name central

    echo "Installing and activating Conda environment"
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -O miniconda-39.sh
    bash miniconda-39.sh -fb
    rm miniconda-39.sh

    source ~/miniconda3/etc/profile.d/conda.sh
    conda init --all
    conda create --name central -y python=3.9.15
    conda activate central
    echo "Conda environment installation complete"
    echo
}

install_kubectl
install_docker
install_conda_environment
