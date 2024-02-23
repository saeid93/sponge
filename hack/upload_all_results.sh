#!/bin/bash

function upload_all() {
    gsutil cp -rn ~/malleable_scaler/data/results gs://malleable_scaler/
}

upload_all
