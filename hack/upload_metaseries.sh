#!/bin/bash

function upload_metasereis(){
    gsutil cp -rn ~/sponge/data/results/final/metaseries/$SERIES gs://sponge/results/final/metaseries/$SERIES
}

upload_metasereis
