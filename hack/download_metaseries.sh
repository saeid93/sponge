#!/bin/bash

# considering that you have already connected to the object storage
function download_metaseries(){
    gsutil cp -rn  gs://malleable-scaler/results/final/metaseries/$SERIES ~/malleable-scaler/data/results/final/metaseries/
}

download_metaseries