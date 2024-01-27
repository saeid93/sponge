#!/bin/bash

function upload_profiling_series(){
    gsutil cp -rn ~/malleable_scaler/data/results/profiling/nodes/series/$SERIES gs://malleable_scaler/results/profiling/nodes/series/$SERIES
}

upload_profiling_series
