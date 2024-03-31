#!/bin/bash

function download_profiling_series(){
    gsutil cp -rn gs://sponge/results/profiling/nodes/series/$SERIES ~/sponge/data/results/profiling/nodes/series/
}

download_profiling_series
sponge/data/results/results/profiling/nodes/series