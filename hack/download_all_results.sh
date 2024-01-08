#!/bin/bash

# considering that you have already connected to the object storage
function download_all(){
    gsutil cp -rn gs://malleable-scaler/results ~/malleable-scaler/data
}

download_all