#!/bin/bash

function download_profiling_series(){
    gsutil cp -rn gs://malleable-scaler/results/profiling/nodes/series/$SERIES ~/malleable-scaler/data/results/profiling/nodes/series/
}

download_profiling_series
malleable-scaler/data/results/results/profiling/nodes/series