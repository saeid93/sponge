#!/bin/bash

# considering that you have already connected to the object storage
function download_all(){
    gsutil cp -rn gs://malleable_scaler/results ~/malleable_scaler/data
}

download_all