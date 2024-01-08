#!/bin/bash

function upload_all() {
    gsutil cp -rn ~/malleable-scaler/data/results gs://malleable-scaler/
}

upload_all
