#!/bin/bash

# considering that you have already connected to the object storage
function download_metaseries(){
    gsutil cp -rn  gs://ipa-results/results/final/metaseries/$SERIES ~/ipa-private/data/results/final/metaseries/
}

download_metaseries