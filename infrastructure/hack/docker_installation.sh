#!/bin/bash

function install_docker() {
    echo "Install Docker"
    sudo apt-get remove -y docker docker-engine docker.io containerd runc
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo groupadd docker
    sudo usermod -aG docker $USER
    # newgrp docker
    sudo systemctl enable docker.service
    sudo systemctl enable containerd.service
    rm get-docker.sh
    echo "End Install Docker"
    echo
}

echo "Running docker script"
install_docker