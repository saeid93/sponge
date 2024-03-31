#!/bin/bash

# considering that you have already connected to the object storage
function download_all(){
    gsutil cp -rn gs://sponge/results ~/sponge/data
}

download_all