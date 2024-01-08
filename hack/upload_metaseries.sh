#!/bin/bash

function upload_metasereis(){
    gsutil cp -rn ~/malleable-scaler/data/results/final/metaseries/$SERIES gs://malleable-scaler/results/final/metaseries/$SERIES
}

upload_metasereis
