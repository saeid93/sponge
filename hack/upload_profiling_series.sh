#!/bin/bash

function upload_profiling_series(){
    gsutil cp -rn ~/malleable-scaler/data/results/profiling/nodes/series/$SERIES gs://malleable-scaler/results/profiling/nodes/series/$SERIES
}

upload_profiling_series
