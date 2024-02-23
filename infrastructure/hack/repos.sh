#!/bin/bash

# Update and install necessary packages
function install_packages() {
    echo "Updating packages"
    sudo apt-get update -y
    echo "Installing packages"
    sudo apt-get install ffmpeg libsm6 libxext6 iperf -y
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

# Install project based on $(PROJECT)
function install_project() {
    case "${PROJECT}" in
        ipa)
            echo "Installing ipa-private requirements"
            cd ~
            git clone https://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@github.com/reconfigurable-ml-pipeline/ipa-private.git
            cd ipa-private
            pip install -r requirements.txt
            cd ..
            echo "ipa-private requirements installation complete"
            echo
            ;;
        inference_x)
            echo "Installing inference_x requirements"
            cd ~
            git clone https://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@github.com/reconfigurable-ml-pipeline/inference_x.git
            cd inference_x
            pip install -r requirements.txt
            cd ..
            echo "inference_x requirements installation complete"
            echo
            ;;
        malleable-scaler)
            echo "Installing malleable-scaler requirements"
            cd ~
            git clone https://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@github.com/saeid93/malleable-scaler.git
            cd malleable-scaler
            pip install -r requirements.txt
            cd ..
            echo "malleable-scaler requirements installation complete"
            echo
            ;;
        all)  # New option to install both projects
            echo "Installing ipa-private requirements"
            cd ~
            git clone https://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@github.com/reconfigurable-ml-pipeline/ipa-private.git
            cd ipa-private
            pip install -r requirements.txt
            cd ..
            echo "ipa-private requirements installation complete"
            echo

            echo "Installing inference_x requirements"
            cd ~
            git clone https://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@github.com/reconfigurable-ml-pipeline/inference_x.git
            cd inference_x
            pip install -r requirements.txt
            cd ..
            echo "inference_x requirements installation complete"
            echo

            echo "Installing malleable-scaler requirements"
            cd ~
            git clone https://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@github.com/saeid93/malleable-scaler.git
            cd malleable-scaler
            pip install -r requirements.txt
            cd ..
            echo "malleable-scaler requirements installation complete"
            echo
            ;;
        *)
            echo "Invalid PROJECT name. It should be either 'ipa', 'inference_x', 'malleable-scaler' or 'all'."
            exit 1
            ;;
    esac
}


# Install load_tester
function install_load_tester() {
    echo "Installing load testetr"
    cd ~
    git clone https://github.com/reconfigurable-ml-pipeline/load_tester.git
    cd load_tester
    git checkout saeed
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

if [ -z "${PROJECT}" ]; then
    echo "You must provide PROJECT name: PROJECT=<project name>"
    exit 1
fi

if [ "${PROJECT}" != "ipa" ] && [ "${PROJECT}" != "inference_x" ] && [ "${PROJECT}" != "all" ]; then
    echo "Invalid PROJECT name. It should be either 'ipa' or 'inference_x' or 'all'."
    exit 1
fi

install_packages
install_conda_environment
install_custom_mlserver
install_project
install_load_tester
install_yolov5
