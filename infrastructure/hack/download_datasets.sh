#!/bin/bash

function make_dataset_folder(){
    mkdir ~/datasets
}

function download_imagenet(){
    mkdir ~/datasets/imagenet
    gsutil cp -rn gs://inference-models-datasets/ILSVRC2012_devkit_t12.tar.gz ~/datasets/imagenet
    gsutil cp -rn gs://inference-models-datasets/ILSVRC2012_img_val.tar ~/datasets/imagenet
}

make_dataset_folder
download_imagenet