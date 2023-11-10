#!/bin/bash

# considering that you have already connected to the object storage
function download_all(){
    gsutil cp -rn gs://ipa-results/results ~/ipa-private/data
}

download_all