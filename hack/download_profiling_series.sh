#!/bin/bash

function download_profiling_series(){
    gsutil cp -rn gs://malleable_scaler/results/profiling/nodes/series/$SERIES ~/malleable_scaler/data/results/profiling/nodes/series/
}

download_profiling_series
malleable_scaler/data/results/results/profiling/nodes/series