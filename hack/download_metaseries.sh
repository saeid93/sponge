#!/bin/bash

# considering that you have already connected to the object storage
function download_metaseries(){
    gsutil cp -rn  gs://sponge/results/final/metaseries/$SERIES ~/sponge/data/results/final/metaseries/
}

download_metaseries