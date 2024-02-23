#!/bin/bash

function upload_metasereis(){
    gsutil cp -rn ~/malleable_scaler/data/results/final/metaseries/$SERIES gs://malleable_scaler/results/final/metaseries/$SERIES
}

upload_metasereis
