#!/bin/bash

# Update and install necessary packages
function install_packages() {
    echo "Updating packages"
    sudo apt-get update -y
    echo "Installing packages"
    sudo apt-get install ffmpeg libsm6 libxext6 iperf zip unzip -y
    echo "Packages installation complete"
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

# Install customized MLServer
function install_custom_mlserver() {
    echo "Installing the customized MLServer"
    cd ~
    git clone https://github.com/saeid93/MLServer.git
    cd MLServer
    git checkout configure-custom-1
    make install-dev
    cd ..
    echo "MLServer installation complete"
    echo
}

# Install sponge requirements
function install_project() {
    echo "Installing sponge requirements"
    cd ~/sponge
    pip install -r requirements.txt
    cd ..
    echo "sponge requirements installation complete"
    echo
}


# Install load_tester
function install_load_tester() {
    echo "Installing load testetr"
    cd ~
    git clone https://github.com/reconfigurable-ml-pipeline/load_tester.git
    cd load_tester
    git checkout open-type-parameters
    pip install -e .
    cd ..
    echo "load tester installation complete"
    echo
}

# Install yolov5
function install_yolov5() {
    echo "Installing yolov5"
    cd ~
    git clone https://github.com/saeid93/yolov5.git
    cd yolov5
    git checkout ipa
    cd ..
    echo "yolov5 installation complete"
    echo
}


install_packages
install_conda_environment
install_custom_mlserver
install_project
install_load_tester
install_yolov5
