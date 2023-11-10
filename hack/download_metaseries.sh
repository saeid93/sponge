#!/bin/bash

# considering that you have already connected to the object storage
function download_all(){
    gsutil cp -rn  gs://ipa-results/results/final/metaseries/$SERIES ~/ipa-private/data/results/final/metaseries/$SERIES
}

download_all